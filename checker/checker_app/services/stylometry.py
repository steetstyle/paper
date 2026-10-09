"""Rule-based stylometry: features that need no model.

This layer always runs, even with no network and no downloaded weights, which
matters because it is the only signal that works on a laptop with nothing
installed. It produces :class:`~checker_app.domain.models.StyleStats` per sentence:

* lexical: word length, type/token ratio, hapax ratio, lexical density, digit
  ratio, uppercase ratio
* punctuation: comma density (dense, parallel clauses read as generated)
* register: contraction/slang rate - an unblemished sentence in an informal
  document is a mild signal in itself
* self repetition: repeated 3-grams inside one sentence
* typography: em dashes, curly quotes, markdown bold, bullets
* lexicon hits from :mod:`checker_app.services.lexicon`

Style hits carry weights that may be negative (``human_marker``): first person
and slang pull the score down.

Document level adds two things that are *context*, not verdicts:
:class:`LexicalRichness` (the strongest feature area in the ablation literature,
with an unstable sign) and :class:`LengthConfound` (word count alone reaches
ROC-AUC 0.68).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import stdev

from checker_app.config import StylometrySettings
from checker_app.domain.enums import BlockType, Severity
from checker_app.domain.models import Sentence, StyleHit, StyleStats
from checker_app.domain.text import Token, fold_text
from checker_app.services.lexicon import Rule, build_rules

__all__ = [
    "StylometryService",
    "DocumentStyleStats",
    "LexicalRichness",
    "LengthConfound",
    "LEXICAL_RICHNESS_DIRECTION",
    "LEXICAL_RICHNESS_SOURCE",
]

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_EM_DASH_RE = re.compile(r"—|–")
_CURLY_QUOTE_RE = re.compile(r"[“”„«»]")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
_BULLET_RE = re.compile(r"[•▪●·]")
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")
_CONTRACTION_RE = re.compile(r"\b\w{2,}(?:n't|'re|'ve|'ll|'d|'m|'s)\b", re.IGNORECASE)
_SLANG_RE = re.compile(
    r"\b(?:gonna|wanna|kinda|gotta|yeah|stuff|guys|a lot|super)\b", re.IGNORECASE
)
_TR_SLANG_RE = re.compile(r"\b(?:ya|bak|valla|abi|reis|patron|hocam)\b", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:]| ve | ile | and | but | ancak | ya da ")

#: Closed-class items only, folded to the same shape as ``Token.norm``. Kept
#: short on purpose: a long list turns "lexical density" into a measurement of
#: the list's own completeness.
_FUNCTION_WORDS = frozenset(
    {
        # Turkish
        "acaba", "alti", "ama", "ancak", "arada", "artik", "asla", "az", "bazi",
        "bazı", "belki", "ben", "bile", "bir", "biraz", "birçok", "biz", "bu",
        "burada", "böyle", "da", "daha", "de", "defa", "diye", "diğer", "elbette",
        "en", "fakat", "gibi", "hem", "hep", "hepsi", "her", "hiç", "için", "ile",
        "ise", "iyi", "kadar", "kez", "ki", "kim", "mi", "mı", "mu", "mü", "nasıl",
        "ne", "neden", "nerede", "nereye", "niçin", "niye", "o", "olan", "olarak",
        "oldu", "olmadı", "olmadığı", "olmak", "olması", "olmayan", "olmaz",
        "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on", "ona", "ondan",
        "onlar", "onlardan", "onları", "onların", "onu", "onun", "orada", "oysa",
        "öyle", "pek", "rağmen", "sadece", "sen", "siz", "sonra", "şey", "şeyden",
        "şeyi", "şeyler", "şöyle", "şu", "tarafından", "tüm", "üzere", "ve", "veya",
        "ya", "yani", "yapmak", "yaptı", "yaptığı", "yapılan", "yapılması",
        "yapıyor", "yine", "yoksa", "zaten",
        # English
        "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "did", "do", "does", "each",
        "few", "for", "from", "further", "had", "has", "have", "he", "here", "hers", "him", "his", "how", "however", "i", "if", "in", "into",
        "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no",
        "nor", "not", "now", "of", "off", "once", "only", "or", "other",
        "our", "ours", "out", "over", "own", "same", "she", "should", "so",
        "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with",
        "would", "you", "your",
    }
)

#: The measured sign of the lexical-richness trio - and the reason it is reported
#: as context rather than scored as a tell. Measured over 64,304 documents
#: (16,076 human news articles x four LLM roles), Cheng vd., arXiv:2410.14259:
#:
#: =============  ==================================
#: role           vocabulary richness (0-1)
#: =============  ==================================
#: human author   0.59 +/- 0.07
#: LLM polisher   0.61 +/- 0.07   <- *above* the human baseline
#: LLM creator    0.52 +/- 0.06
#: LLM extender   0.51 +/- 0.06
#: =============  ==================================
#:
#: Light AI editing raises lexical richness above the human value; generation
#: lowers it. One threshold cannot mean "written by a machine", because the two
#: AI roles push in opposite directions.
LEXICAL_RICHNESS_DIRECTION: dict[str, tuple[float, float]] = {
    "human_author": (0.59, 0.07),
    "llm_polisher": (0.61, 0.07),
    "llm_creator": (0.52, 0.06),
    "llm_extender": (0.51, 0.06),
}

LEXICAL_RICHNESS_SOURCE = (
    "Cheng vd., arXiv:2410.14259 (WWW 2025): 16.076 insan haberi x 4 LLM rolu = "
    "64.304 belge. Ablation: El Attar vd., arXiv:2606.04177 (284 ozellik, 27 LLM, "
    "10 alan) - leksikal zenginligin dusurulmesi alan ici 13.1, alan disi 27.7 "
    "F1 puani kaybettirir; iki ozelligin (hapax, TTR) alan disindaki tek basina "
    "AUC degeri 0.97 (1.8M makine uretimi; arXiv:2603.18482)."
)

#: The Turkish measurement, and it inverts the naive reading. Renklier &
#: Sarıtaş (2026, DOI 10.28948/ngumuh.1930411) measured mean TTR over 3,000 human
#: and 9,000 LLM Turkish texts:
#:
#: ===================  =======
#: source               mean TTR
#: ===================  =======
#: human                **0.756**
#: GPT-4o              0.753
#: Gemini 2.5 Flash    0.780
#: DeepSeek V3          0.808
#: Claude Sonnet 4.6    0.808
#: GPT-3.5-turbo        0.738
#: Llama 3.3 70B        **0.709**
#: ===================  =======
#:
#: The human value sits *inside* the AI range [0.709, 0.808] - Claude and
#: DeepSeek are **more** lexically diverse than Turkish humans. So TTR cannot
#: separate the classes in Turkish at all, which is why this tool reports the
#: trio as context and never scores it.
LEXICAL_RICHNESS_TURKISH: dict[str, float] = {
    "human": 0.756,
    "gpt-4o": 0.753,
    "gemini-2.5-flash": 0.780,
    "deepseek-v3": 0.808,
    "claude-sonnet-4.6": 0.808,
    "gpt-3.5-turbo": 0.738,
    "llama-3.3-70b": 0.709,
}

LEXICAL_RICHNESS_TURKISH_NOTE = (
    "Türkçede TTR sınıfları ayırmaz: insan 0.756, LLM aralığı 0.709-0.808 — insan "
    "değeri aralığın İÇİNDE, Claude ve DeepSeek insanlardan daha zengin. Bu yüzden "
    "Türkçede leksikal zenginlik AI işareti olarak kullanılamaz (DOI "
    "10.28948/ngumuh.1930411)."
)

#: Turkish human text is also measurably **longer** than Turkish LLM text, which
#: makes length a sharper confound here than the general English measurement
#: suggests. And human Turkish shows the *highest* subword fragmentation of the
#: set (1.522 tokens/word, above every LLM), so tokens-per-word is not a usable
#: AI signal either - it points the opposite way from the naive expectation.
LENGTH_TURKISH_HUMAN_WORDS = 234.7
LENGTH_TURKISH_LLM_WORDS = (141.2, 184.8)
LENGTH_TURKISH: dict[str, object] = {
    "human_words": LENGTH_TURKISH_HUMAN_WORDS,
    "llm_words": list(LENGTH_TURKISH_LLM_WORDS),
    "human_tokens_per_word": 1.522,
}


def _is_numeric(text: str) -> bool:
    return bool(text) and all(ch.isdigit() or ch in ".," for ch in text)


@dataclass(frozen=True, slots=True)
class DocumentStyleStats:
    """Document level stylometry, used for relative (z-score) features."""

    mean_length: float = 0.0
    std_length: float = 0.0
    mean_word_length: float = 0.0
    informal_rate: float = 0.0
    sentence_count: int = 0
    word_count: int = 0
    lexical_richness: LexicalRichness | None = None
    length_confound: LengthConfound | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_sentence_length": round(self.mean_length, 2),
            "std_sentence_length": round(self.std_length, 2),
            "mean_word_length": round(self.mean_word_length, 2),
            "informal_rate": round(self.informal_rate, 4),
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "lexical_richness": (
                self.lexical_richness.to_dict() if self.lexical_richness else None
            ),
            "length_confound": self.length_confound.to_dict() if self.length_confound else None,
        }


@dataclass(frozen=True, slots=True)
class LexicalRichness:
    """The lexical-richness trio, computed on the whole document.

    Sentence-level TTR and hapax are unstable because a sentence has too few
    tokens for either to mean anything; the document has enough. Computed once
    and reported as *context*, never as a verdict - see
    :data:`LEXICAL_RICHNESS_DIRECTION` for why the sign is not fixed.
    """

    type_token_ratio: float
    hapax_ratio: float
    lexical_density: float
    word_count: int
    sentence_count: int
    sentence_ttr_spread: float
    """SD of per-sentence TTR. A document whose sentences vary more is built
    from more than one author or register; one that does not was likely written
    in a single pass, which is the only thing this number honestly supports."""

    def to_dict(self) -> dict[str, object]:
        closest, label = self.closest_role()
        return {
            "type_token_ratio": round(self.type_token_ratio, 4),
            "hapax_ratio": round(self.hapax_ratio, 4),
            "lexical_density": round(self.lexical_density, 4),
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "sentence_ttr_spread": round(self.sentence_ttr_spread, 4),
            "closest_measured_role": label,
            "closest_measured_value": closest,
            "measured_reference": {
                role: {"mean": mean, "sd": sd}
                for role, (mean, sd) in LEXICAL_RICHNESS_DIRECTION.items()
            },
            "source": LEXICAL_RICHNESS_SOURCE,
            "reading": (
                "Leksikal zenginlik AI işareti olarak tek yönlü kullanılamaz: hafif "
                "AI düzenlemesi insan değerinin ÜSTÜNE çıkarır, üretim altına indirir "
                "(arXiv:2410.14259). Bu yüzden puan değil, bağlam olarak raporlanır. "
                + LEXICAL_RICHNESS_TURKISH_NOTE
            ),
            "turkish_reference": LEXICAL_RICHNESS_TURKISH,
        }

    def closest_role(self) -> tuple[float, str]:
        """Nearest measured reference role for the blended richness value.

        TTR, hapax ratio and lexical density are averaged after normalising each
        to the 0-1 range the reference study reports. It is a coarse comparison
        - the reference is news prose and this is a thesis - and is labelled as
        such wherever it is shown.
        """
        blended = (
            self.type_token_ratio + self.hapax_ratio + self.lexical_density
        ) / 3
        best = min(
            LEXICAL_RICHNESS_DIRECTION.items(),
            key=lambda kv: abs(kv[1][0] - blended),
        )
        return blended, best[0]


@dataclass(frozen=True, slots=True)
class LengthConfound:
    """Word count is a confounder, and this is how strong that is.

    Kumar vd. (arXiv:2609.26687, HICSS 2027) measured a **cross-validated
    ROC-AUC of 0.68 from word count alone** on academic prose, and reported that
    GPT-assisted writing is measurably *shorter*: 620 +/- 198 words versus
    515 +/- 155, paired t(89) = 6.46, p < .001, longer in 72 of 90 pairs.

    So a short document is not neutral evidence. This object says which regime
    the document sits in, rather than letting the reader assume it does not
    matter. Sentence length, notably, is *not* a usable signal: the same study
    found sentence-length variability alone performs near chance, and mean
    sentence length did not differ between conditions (27.2 vs 27.3, p = .97).
    """

    word_count: int
    sentence_count: int
    regime: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "regime": self.regime,
            "note": self.note,
            "measured": {
                "word_count_only_auroc": 0.68,
                "human_words": "620 +/- 198",
                "assisted_words": "515 +/- 155",
                "paired_t": "t(89)=6.46, p<.001",
                "longer_in_pairs": "72/90",
            },
            "source": "arXiv:2609.26687 (Kumar vd., HICSS 2027)",
            "turkish_measured": {
                "human_words": LENGTH_TURKISH["human_words"],
                "llm_words_range": LENGTH_TURKISH["llm_words"],
                "source": "DOI 10.28948/ngumuh.1930411",
                "excluded": (
                    "İnsan Türkçe metinde alt-kelime parçalanması en YÜKSEK değeri "
                    f"gösterir ({LENGTH_TURKISH['human_tokens_per_word']} token/kelime), "
                    "her LLM'nin üzerinde. Token/kelime oranı da AI sinyali "
                    "olarak kullanılamaz."
                ),
            },
            "excluded": (
                "Cümle uzunluğu değişkenliği tek başına şansa yakın; ortalama cümle "
                "uzunluğu koşullar arasında fark göstermedi (27.2 / 27.3, p=.97). "
                "Bu yüzden ağırlıklandırılmaz."
            ),
        }


class StylometryService:
    """Compute style features for sentences and for a document as a whole."""

    def __init__(
        self,
        settings: StylometrySettings | None = None,
        rules: Sequence[Rule] | None = None,
    ) -> None:
        self._settings = settings or StylometrySettings()
        self._rules = (
            tuple(rules)
            if rules is not None
            else build_rules(tuple(self._settings.lexicon_languages))
        )

    # ------------------------------------------------------------------ public
    def document_stats(
        self, sentences: Sequence[Sentence], tokens: Sequence[Token] = ()
    ) -> DocumentStyleStats:
        lengths = [s.word_count for s in sentences if s.word_count]
        words = [s.word_count for s in sentences if s.word_count]
        if not lengths:
            return DocumentStyleStats()
        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        informal = [self._informal_rate(_surface_tokens(s)) for s in sentences]
        return DocumentStyleStats(
            mean_length=mean,
            std_length=variance**0.5,
            mean_word_length=(
                sum(len(w) for s in sentences for w in _WORD_RE.findall(s.text))
                / max(1, sum(words))
            ),
            informal_rate=sum(informal) / len(informal) if informal else 0.0,
            sentence_count=len(lengths),
            word_count=len(tokens) or sum(words),
            lexical_richness=self._lexical_richness(sentences, tokens),
            length_confound=self._length_confound(len(tokens) or sum(words), len(lengths)),
        )

    def _lexical_richness(
        self, sentences: Sequence[Sentence], tokens: Sequence[Token]
    ) -> LexicalRichness | None:
        """The trio, on the whole document.

        Needs a real token stream: with fewer than 200 tokens TTR and hapax are
        noise, and reporting a number for a two-sentence fixture would be the
        kind of false precision this tool exists to avoid.
        """
        usable = [t for t in tokens if t.norm and not _is_numeric(t.text)]
        if len(usable) < 200:
            return None
        per_sentence = [
            self._type_token_ratio(_surface_tokens(s))
            for s in sentences
            if s.word_count >= 20
        ]
        spread = (
            round(stdev(per_sentence), 4)
            if len(per_sentence) > 1
            else 0.0
        )
        return LexicalRichness(
            type_token_ratio=self._type_token_ratio(usable),
            hapax_ratio=self._hapax_ratio(usable),
            lexical_density=self._lexical_density(usable),
            word_count=len(usable),
            sentence_count=len(per_sentence),
            sentence_ttr_spread=spread,
        )

    @staticmethod
    def _length_confound(word_count: int, sentence_count: int) -> LengthConfound:
        """Which length regime the document sits in.

        Thresholds bracket the measured study condition (515-620 words). Below
        it the comparison in that study is not available at all; far above it,
        a single flagged sentence moves the share very little, which is the safe
        direction.
        """
        if word_count < 200:
            regime = "cok_kisa"
            note = (
                f"Belge {word_count} kelime: kelime sayısının tek başına ROC-AUC 0.68 "
                "olduğu ölçülen aralığın çok altında. Bu hacimde belge düzeyi "
                "oranlar kararsızdır; yalnız cümle düzeyi okunmalıdır."
            )
        elif word_count < 800:
            regime = "calisma_kosuluna_yakin"
            note = (
                f"Belge {word_count} kelime, ölçülen çalışmanın koşul aralığında "
                "(insan 620 +/- 198, AI destekli 515 +/- 155; t(89)=6.46, p<.001). "
                "Bu aralıkta kısa bir belge neden-sonuç olarak yorumlanamaz, "
                "çünkü kısalık tek başına sinyal üretiyor."
            )
        else:
            regime = "uzun"
            note = (
                f"Belge {word_count} kelime: kısalık karıştırıcısı zayıflıyor ve "
                "yapı sinyalleri (söylem bağlaçları) biriktiği için tersine güçleniyor "
                "(arXiv:2402.10586: 1/3/5 üretilmiş paragrafta F1 0.43 → 0.71)."
            )
        return LengthConfound(
            word_count=word_count,
            sentence_count=sentence_count,
            regime=regime,
            note=note
            + (
                f" Türkçede etki daha keskin: insan metni ortalama "
                f"{LENGTH_TURKISH_HUMAN_WORDS} kelime, LLM metni "
                f"{LENGTH_TURKISH_LLM_WORDS[0]}-{LENGTH_TURKISH_LLM_WORDS[1]} kelime, "
                "yani %21-40 daha kısa (DOI 10.28948/ngumuh.1930411)."
            ),
        )

    def analyze(self, sentence: Sentence, tokens: Sequence[Token]) -> StyleStats:
        text = sentence.text
        words = [t.text for t in tokens]
        word_count = len(words)
        char_count = len(text.strip())
        stats = StyleStats(
            word_count=word_count,
            char_count=char_count,
            avg_word_length=(sum(len(w) for w in words) / word_count if word_count else 0.0),
            comma_density=self._comma_density(text, word_count),
            type_token_ratio=self._type_token_ratio(tokens),
            hapax_ratio=self._hapax_ratio(tokens),
            lexical_density=self._lexical_density(tokens),
            informal_rate=self._informal_rate(tokens),
            digits_ratio=self._ratio(text, r"\d"),
            upper_ratio=self._ratio(text, r"[A-ZÇĞİÖŞÜ]"),
        )
        if not self._settings.enabled or word_count < self._settings.min_sentence_words:
            return stats
        if sentence.location.block_type in {
            BlockType.CODE,
            BlockType.MATH,
            BlockType.TABLE,
        }:
            # Features are still reported, but "this code has no contractions"
            # is not a finding about authorship.
            return stats

        hits: list[StyleHit] = []
        hits.extend(self._lexicon_hits(text))
        hits.extend(self._typography_hits(text, sentence.location.block_type))
        hits.extend(self._register_hits(text, stats))
        hits.extend(self._repetition_hits(tokens))
        return StyleStats(
            word_count=stats.word_count,
            char_count=stats.char_count,
            avg_word_length=stats.avg_word_length,
            comma_density=stats.comma_density,
            type_token_ratio=stats.type_token_ratio,
            hapax_ratio=stats.hapax_ratio,
            lexical_density=stats.lexical_density,
            informal_rate=stats.informal_rate,
            digits_ratio=stats.digits_ratio,
            upper_ratio=stats.upper_ratio,
            hits=tuple(sorted(hits, key=lambda h: -abs(h.weight))),
        )

    # ----------------------------------------------------------------- private
    def _lexicon_hits(self, text: str) -> list[StyleHit]:
        hits: list[StyleHit] = []
        for rule in self._rules:
            evidence = rule.matches(text)
            if not evidence:
                continue
            hits.append(
                StyleHit(
                    code=rule.code,
                    severity=rule.severity,
                    weight=rule.weight,
                    detail=_RULE_DETAIL.get(rule.code, rule.code),
                    evidence=" | ".join(evidence),
                )
            )
        return hits

    def _typography_hits(self, text: str, block_type: BlockType) -> list[StyleHit]:
        hits: list[StyleHit] = []
        if block_type in {BlockType.CODE, BlockType.MATH, BlockType.TABLE}:
            return hits
        marks: list[tuple[str, str, float, Severity]] = []
        if _EM_DASH_RE.search(text):
            marks.append(("em_dash", "Uzun tire (—) kullanımı", 0.06, Severity.LOW))
        if _CURLY_QUOTE_RE.search(text):
            marks.append(("typographic_quotes", "Tipografik tırnak kullanımı", 0.04, Severity.LOW))
        if _MARKDOWN_BOLD_RE.search(text):
            marks.append(("markdown_bold", "Markdown kalın vurgu", 0.05, Severity.LOW))
        if _BULLET_RE.search(text):
            marks.append(("bullet_glyph", "Madde işareti glifi", 0.04, Severity.LOW))
        if _EMOJI_RE.search(text):
            marks.append(("emoji", "Emoji", 0.07, Severity.LOW))
        for code, detail, weight, severity in marks:
            hits.append(
                StyleHit(
                    code=code,
                    severity=severity,
                    weight=weight,
                    detail=detail,
                    evidence=_first_match(text, code),
                )
            )
        return hits

    def _register_hits(self, text: str, stats: StyleStats) -> list[StyleHit]:
        """A perfectly clean sentence inside an informal document."""
        if stats.word_count < 8 or stats.informal_rate > 0:
            return []
        return [
            StyleHit(
                code="register_too_clean",
                severity=Severity.LOW,
                weight=0.07,
                detail="Gövde dili tamamen resmî, kısaltma veya üslup kırılması yok",
                evidence="",
            )
        ]

    def _repetition_hits(self, tokens: Sequence[Token]) -> list[StyleHit]:
        if len(tokens) < 10:
            return []
        seen: set[tuple[str, str, str]] = set()
        repeated = 0
        for i in range(len(tokens) - 2):
            gram = (tokens[i].norm, tokens[i + 1].norm, tokens[i + 2].norm)
            if gram in seen:
                repeated += 1
            seen.add(gram)
        if repeated < 2:
            return []
        return [
            StyleHit(
                code="self_repetition",
                severity=Severity.LOW,
                weight=min(0.10, 0.04 * repeated),
                detail="Cümle içinde yinelenen üçlü kelime grupları",
                evidence=f"{repeated} yineleme",
            )
        ]

    @staticmethod
    def _comma_density(text: str, word_count: int) -> float:
        if word_count < 6:
            return 0.0
        clauses = len(_CLAUSE_SPLIT_RE.findall(text)) + 1
        return round(text.count(",") / max(1, clauses), 4)

    @staticmethod
    def _type_token_ratio(tokens: Sequence[Token]) -> float:
        if not tokens:
            return 0.0
        return round(len({t.norm for t in tokens}) / len(tokens), 4)

    @staticmethod
    def _hapax_ratio(tokens: Sequence[Token]) -> float:
        """Word types occurring exactly once, over all tokens.

        Part of the lexical-richness trio that dominates the ablation study in
        arXiv:2606.04177. It is *not* scored as a tell - see
        :data:`LEXICAL_RICHNESS_DIRECTION` for why the sign is unstable.
        """
        if not tokens:
            return 0.0
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token.norm] = counts.get(token.norm, 0) + 1
        hapax = sum(1 for count in counts.values() if count == 1)
        return round(hapax / len(tokens), 4)

    @staticmethod
    def _lexical_density(tokens: Sequence[Token]) -> float:
        """Content-word share: tokens that are not function words.

        The third member of the trio. The stop list is deliberately small and
        folded, because a Turkish function word list that is merely long makes
        the feature a proxy for the stop list's own completeness.
        """
        if not tokens:
            return 0.0
        content = sum(1 for token in tokens if token.norm not in _FUNCTION_WORDS)
        return round(content / len(tokens), 4)

    @staticmethod
    def _informal_rate(tokens: Sequence[Token]) -> float:
        if not tokens:
            return 0.0
        surface = " ".join(t.text for t in tokens)
        hits = len(_CONTRACTION_RE.findall(surface)) + len(_SLANG_RE.findall(surface))
        hits += len(_TR_SLANG_RE.findall(fold_text(surface, strip_diacritics=False)))
        return round(min(1.0, hits / max(1, len(tokens)) * 4), 4)

    @staticmethod
    def _ratio(text: str, pattern: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        return round(len(re.findall(pattern, stripped)) / len(stripped), 4)


def _surface_tokens(sentence: Sentence) -> list[Token]:
    """Cheap token view used for document level statistics only."""
    return [
        Token(
            index=i,
            text=w,
            norm=fold_text(w),
            char_start=0,
            char_end=len(w),
            is_number=w[0].isdigit(),
        )
        for i, w in enumerate(_WORD_RE.findall(sentence.text))
    ]


def _first_match(text: str, code: str) -> str:
    pattern = {
        "em_dash": _EM_DASH_RE,
        "typographic_quotes": _CURLY_QUOTE_RE,
        "markdown_bold": _MARKDOWN_BOLD_RE,
        "bullet_glyph": _BULLET_RE,
        "emoji": _EMOJI_RE,
    }.get(code)
    if pattern is None:
        return ""
    m = pattern.search(text)
    return m.group(0) if m else ""


_RULE_DETAIL: dict[str, str] = {
    "discourse_marker": "Geçiş/diskurs kalıpları (Ayrıca, Furthermore, In conclusion…)",
    "hedge_stack": "Üst üste belirsizlik ifadeleri",
    "intensifier_stack": "Abartılı pekiştireçler",
    "favourite_noun": "Modellerin sık kullandığı sözcükler",
    "presentative_passive": "Sunum dili (…sunulmaktadır, bu makalede)",
    "meta_hedge": "Meta belirsizlik kalıpları",
    "strong_positivity": "Aşırı olumlu değerlendirme dili",
    "human_marker": "Birinci şahıs / günlük dil (insan yazımı lehine)",
}
