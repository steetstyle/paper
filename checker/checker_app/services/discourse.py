"""Discourse-connective profile: the cheapest measured signal with a known direction.

The motivation is RST (Rhetorical Structure Theory) relation distributions, which
are the only *structural* AI-authorship signal with a measured effect size rather
than an intuition behind it. Li, Sheng, Wang, Yang & Cao, "Beyond the Final
Actor: Modeling the Dual Roles of Creator and Editor" (RACE, arXiv:2604.04932,
ACL 2026) measured, over 16 076 human news articles across four LLM roles:

* human creators **over-express Attribution and Temporal** relations;
* LLM creators **over-express Elaboration and Cause** - a flatter hierarchy, in
  which paragraphs elaborate rather than argue from premises.

Two properties make this worth implementing, because most stylometric signals
lack them:

1. **It survives editing.** The same paper measures cosine similarity between a
   text and its variant: human → AI-polished **0.92 ± 0.08**, human → humanized
   0.86 ± 0.12, human → LLM-generated 0.84 ± 0.12. Rhetorical structure is
   comparatively stable, so the fingerprint points at the human behind a polish
   rather than at the polishing.
2. **It is lexicon-only.** A full RST parse is not available offline for Turkish;
   the four relation classes are countable from connectives, which is what this
   module does. The honest cost: this is a proxy for RST relation distribution,
   not the distribution itself, and connective counts are also what a
   non-native writer or a translation produces.

Where the gain from discourse features is real, and where it is not:

* Adding discourse-motif features to a graph model moved HC3 F1 **0.67 → 0.73**
  (Kim et al., "Threads of Subtlety", arXiv:2402.10586, ACL 2024), and to a
  long-context encoder HC3 **0.97 → 0.98** and OOD-paraphrase **0.60 → 0.62**.
* The motifs *alone* are weak: F1 **0.55 / 0.58 / 0.49 / 0.61** across four test
  sets. So this module reports a profile, never a verdict, and its weight in the
  fused score stays small.
* Discourse features **do not degrade with document length**: F1 rises with
  length (LF 0.30 → 0.45 at 1/3/5 generated paragraphs; LF+Motifs 0.43 → 0.71 on
  TenPageStories, up to ~10 A4 pages). Unlike token-level perplexity, which is
  diluted by long context, structure accumulates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from checker_app.domain.text import Token

__all__ = ["DiscourseProfile", "RELATION_DIRECTION", "discourse_profile"]

#: The measured direction. Only ``elaboration`` and ``temporal``/``attribution``
#: have a published sign; ``cause`` is included because the same paper reports
#: LLM over-expression of it. Anything without a sign is deliberately absent.
RELATION_DIRECTION: dict[str, str] = {
    "attribution": "human",
    "temporal": "human",
    "elaboration": "llm",
    "cause": "llm",
}

# Connectives, not discourse-marker *frequency*: a frequency study on Chinese
# short comments reached F1 91.8 (2025.ccl-1.64) but that is not academic
# prose, so only the relation mapping from RACE is used here.
_CONNECTIVES: Mapping[str, tuple[str, ...]] = {
    "attribution": (
        "because", "since", "therefore", "thus", "hence", "so", "consequently",
        "neden", "çünkü", "bu yüzden", "dolayısıyla", "bu nedenle", "o halde",
        "sonuç olarak", "başka bir deyişle",
    ),
    "temporal": (
        "first", "firstly", "second", "secondly", "third", "thirdly", "finally",
        "then", "next", "subsequently", "afterwards", "previously", "initially",
        "ilk olarak", "ikinci olarak", "üçüncü olarak", "son olarak", "ardından",
        "daha sonra", "öncelikle", "nihayetinde", "başlangıçta",
    ),
    "elaboration": (
        "furthermore", "moreover", "additionally", "in addition", "also",
        "similarly", "likewise", "that is", "in other words", "for example",
        "for instance", "namely", "specifically", "besides", "in fact",
        "ayrıca", "bunun yanı sıra", "üstelik", "ek olarak", "aynen",
        "benzer şekilde", "örneğin", "şöyle ki", "yani", "özellikle", "kısacası",
        "diğer yandan", "buna karşılık",
    ),
    "cause": (
        "leads to", "led to", "leads", "results in", "resulted in", "results",
        "due to", "owing to", "causes", "caused by", "gives rise to",
        "yol açar", "yol açtı", "neden olur", "neden oldu", "sebebiyle",
        "sonucunu doğurur", "doğurdu", "kaynaklanır", "kaynaklanmıştır",
    ),
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + r")(?!\w)",
        re.IGNORECASE,
    )
    for name, terms in _CONNECTIVES.items()
}


@dataclass(frozen=True, slots=True)
class DiscourseProfile:
    """Connective counts per relation class, and the one comparison that means
    something: elaboration against attribution+temporal."""

    counts: Mapping[str, int]
    words: int

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def per_100_words(self) -> dict[str, float]:
        if not self.words:
            return {name: 0.0 for name in _PATTERNS}
        return {name: 100.0 * value / self.words for name, value in self.counts.items()}

    @property
    def elaboration_ratio(self) -> float:
        """Elaboration share of all connectives.

        The published direction is a *relative* one: LLM text elaborates where
        human text attributes and sequences. One number captures it, and it is
        bounded, which matters because raw counts scale with document length.
        """
        if not self.total:
            return 0.0
        return self.counts.get("elaboration", 0) / self.total

    def reading(self) -> str:
        ratio = self.elaboration_ratio
        if not self.total:
            return (
                "Belgede sayılabilir edat bağlacı yok; yapı profili bu metin için "
                "bilgi üretmiyor (kısa bölüm, liste veya alıntı ağırlıklı olabilir)."
            )
        lean = (
            "insan yazar lehine (atıf + sıralama baskın)"
            if ratio < 0.40
            else "LLM lehine (açımlama baskın)"
            if ratio > 0.60
            else "belirgin değil"
        )
        return (
            f"{self.total} bağlacı / {self.words} kelime → 100 kelimede "
            f"{self.per_100_words.get('elaboration', 0):.1f} açımlama, "
            f"{self.per_100_words.get('attribution', 0):.1f} atıf, "
            f"{self.per_100_words.get('temporal', 0):.1f} sıralama. "
            f"Açımlama payı %{ratio * 100:.0f} → {lean}. "
            "Yön RACE (arXiv:2604.04932) ölçümüdür; bu bir hüküm değil, bir bağlamdır."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": dict(self.counts),
            "per_100_words": {k: round(v, 2) for k, v in self.per_100_words.items()},
            "elaboration_ratio": round(self.elaboration_ratio, 3),
            "connectives": self.total,
            "words": self.words,
            "reading": self.reading(),
            "measured_direction": RELATION_DIRECTION,
        }


def discourse_profile(text: str, tokens: Sequence[Token] | None = None) -> DiscourseProfile:
    """Count relation-mapped connectives in ``text``.

    ``tokens`` supplies the word count when available; otherwise it is derived
    from the text, which is what the per-100-words rates are normalised by.
    """
    counts = {name: len(pattern.findall(text)) for name, pattern in _PATTERNS.items()}
    words = len(tokens) if tokens else len(re.findall(r"\w+", text))
    return DiscourseProfile(counts=counts, words=words)