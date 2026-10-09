"""English-language academic baselines: what "normal" looks like in English.

The Turkish pass had a measured AI-presence distribution. This module is its
English counterpart, plus the English style reference values that let a report
say "your lexical diversity is 61, and human English academic prose sits at 93"
instead of leaving the reader to guess.

The single most consequential number is not in here: it is that clean English
dissertations measure **9% ± 6%** similarity while Turkish ones measure
**28.7% ± 11.58** (Mayes 2017, n=360; Toprak 2014, n=600). That lives in
:mod:`checker_app.services.similarity` as ``EnglishBaseline`` because it is a
percentage comparison, not a writing-style reference.

Provenance is explicit on every table. Values are either (a) quoted from a
published source with a DOI, or (b) measured by the research pass on a stated
corpus, and labelled ``original measurement``. Nothing is interpolated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EnglishContext:
    """The English reference bundle, attached only to English documents.

    Kept as one object so the runner has a single field to set, and so the
    language gate is impossible to forget at the point of use.
    """

    presence: Mapping[str, object]
    style: Mapping[str, object]
    genre_note: str = (
        "Dedektörlerin akademik metindeki doğruluğu türe göre değişiyor: "
        "Turnitin insanlıkta 0.86, fen bilimlerinde 0.51 (Hadra vd., 2026, "
        "DOI 10.1007/s40979-026-00213-1). Uzunluğa göre de: 300-330 kelimede "
        "0.87, 450-550 kelimede 0.56 (p=.0149). Bu yüzden İngilizce eşikler "
        "bölüm ve uzunluk bandına göre ayarlanmalıdır; tek belge puanı yanlış olur."
    )
    l1_l2_note: str = (
        "Doğrulanmış intihal vakalarının %82'si İngilizcenin resmî dil olmadığı "
        "ülkelerden (Higgins vd. 2016). Ana dilini İngilizce olmayan bir yazarın "
        "metni insan yazımı olduğu halde düşük leksik çeşitlilik gösterebilir "
        "(A2 seviyesinde MTLD 58 ± 22, L1 akademik medyan 94); bu bir kusur "
        "değil, bir ölçüm farkıdır."
    )
    replication_note: str = (
        "DİKKAT: Liang vd. (2023) L2 yanlış pozitif oranını %61.4 olarak "
        "raporladı (TOEFL-91, 7 dedektör; Originality.AI %75.8, Quil.org %74.7). "
        "Ancak EACL 2026'da aynı veri setinde oran %23.1'e düştü ve Çek "
        "non-native metinlerinde entropinin **yüksek** olduğu görüldü - yön bile "
        "değişti. Bu bulgu sabit bir sabit değil, yazım popülasyonuna göre "
        "kalibrasyon problemidir; bu yüzden araç sabit bir yanlış pozitif oranı "
        "varsaymaz."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "ai_presence": dict(self.presence),
            "style_reference": dict(self.style),
            "genre_and_length_note": self.genre_note,
            "l1_l2_note": self.l1_l2_note,
            "replication_note": self.replication_note,
        }


__all__ = [
    "EnglishAiPresence",
    "EnglishStyleReference",
    "EnglishContext",
    "LLM_WORD_SIGNALS",
    "LLM_AVOIDED_CONNECTIVES",
    "LLM_WORD_SOURCE",
    "PROVENANCE",
]

#: Where each number came from. Kept as one visible place rather than scattered
#: through the docstrings, so a reader can audit the whole module at once.
PROVENANCE = {
    "liang": (
        "Liang vd., arXiv:2404.01268 (COLM 2024) ve Nature Human Behaviour "
        "9(12):2599-2609, 2025, DOI 10.1038/s41562-025-02273-8"
    ),
    "kobak": (
        "Kobak vd., arXiv:2406.07016; Science Advances 2025, "
        "DOI 10.1126/sciadv.adt3813"
    ),
    "thelwall": (
        "Thelwall & Kousha, arXiv:2604.07565 (2026) - 1.25M MDPI full texts, "
        "2021-2025"
    ),
    "hadra": (
        "Hadra vd., Int. J. Educational Integrity 22:4 (2026), "
        "DOI 10.1007/s40979-026-00213-1"
    ),
    "liang2023": (
        "Liang vd., Patterns 4(7):100779 (2023), DOI 10.1016/j.patter.2023.100779"
    ),
    "original": (
        "research pass tarafından ölçüldü: 4.462 pre-LLM arXiv özeti (2015-2019, "
        "742.788 token) + 48 PMC OA tam metin, 113.928 token bölüm metni"
    ),
}


@dataclass(frozen=True, slots=True)
class EnglishAiPresence:
    """How much LLM-modified text English academic writing actually carries.

    The corpus-scale series, from Liang vd. (2024; 2025). A 1,121,912-paper
    corpus for the September 2024 figure:

    ============================  ==========  =============  ============
    venue                         abstracts    introductions  Nov 2022 base
    ============================  ==========  =============  ============
    arXiv Computer Science        **22.5%**    **19.6%**      2.4%
    arXiv Electrical Eng./Systems 18.0%        18.4%          2.9%
    arXiv **Mathematics**         **7.7%**     **4.1%**       2.5%
    Nature portfolio (15 journals) 8.9%        9.4%          3.4%
    ============================  ==========  =============  ============

    Three readings matter for a thesis tool:

    * **Mathematics is the outlier.** Abstract 7.7% and introduction 4.1%, both
      barely above the 2-3% estimator floor. Mathematics is the only venue where
      introductions are *less* modified than abstracts. A mathematics thesis
      read against the CS figure would be flagged on the wrong prior.
    * **Per-section direction is published, the per-section values are not.**
      Liang vd. state that abstracts, introductions, related work and
      conclusions carry more LLM-modified content than **experiment and method**
      sections. The numbers themselves appear only in supplementary figures and
      are UNVERIFIED. The direction is enough to order a review, which is what
      this tool uses it for.
    * **The estimator's own error is < 3.5 percentage points** at population
      level across ground-truth values 0-25% in 5% steps (1,000 bootstrap
      iterations). Any prevalence figure should be read with that band.

    Independent corroboration from a different method and a different corpus:
    Kobak vd. (2025) estimate **>= 13.5%** of PubMed abstracts for 2024 using
    excess-vocabulary frequency rather than a detector, with a range from below
    5% to over 40% across PubMed subcorpora and ~20% in computational fields.
    The best single word moved the estimate 5.2 percentage points ("potential").

    Two operational facts:

    * **Disclosure almost never happens**: 2 of 200 inspected arXiv CS papers
      (1.0%) disclosed LLM use in the writing itself.
    * **LLM-modified peer reviews** run **6.5%-16.9%** (ICLR 2024, NeurIPS 2023,
      CoRL 2023, EMNLP 2023; Liang vd., arXiv:2403.07183).
    """

    arxiv_cs_abstracts: float = 0.225
    arxiv_cs_introductions: float = 0.196
    arxiv_math_abstracts: float = 0.077
    arxiv_math_introductions: float = 0.041
    arxiv_eess_abstracts: float = 0.180
    nature_portfolio_abstracts: float = 0.089
    pre_chatgpt_baseline: float = 0.024
    pubmed_2024_lower_bound: float = 0.135
    estimator_error_pp: float = 3.5
    disclosure_rate: float = 0.010
    """2 of 200 inspected arXiv CS papers."""

    peer_review_range: tuple[float, float] = (0.065, 0.169)
    least_affected_sections: tuple[str, ...] = ("methods", "experiments")
    """Published direction only; the per-section values are not."""

    sample_size: int = 1_121_912
    source: str = PROVENANCE["liang"]
    corroborating_source: str = PROVENANCE["kobak"]
    caveat: str = (
        "Makale/derleme dili, tez değil. Bölüm bazlı oranlar yalnızca yön olarak "
        "yayımlanmıştır; sayısal değerler UNVERIFIED. Tahmin yönteminin kendi "
        f"hata payı <{estimator_error_pp} yüzde puandır."
    )

    def reading(self, share: float) -> str:
        if share < self.pre_chatgpt_baseline:
            verdict = (
                f"%{share * 100:.1f} tahmin yönteminin taban değerinin "
                f"(%{self.pre_chatgpt_baseline * 100:.1f}) altında; bu oran "
                "insan yazımından ayırt edilemiyor."
            )
        elif share < self.arxiv_math_abstracts:
            verdict = (
                f"%{share * 100:.1f} matematik seviyesinde (%7.7) veya altında. "
                "İnsan yazımıyla karışıyor olabilir."
            )
        elif share < self.arxiv_cs_abstracts:
            verdict = (
                f"%{share * 100:.1f} disiplin ortalamasının altında; "
                f"arXiv CS özetleri %22.5, matematik %7.7."
            )
        else:
            verdict = (
                f"%{share * 100:.1f} arXiv CS özet ortalamasının (%22.5) "
                "üstünde — bu, ölçülmüş en yüksek akademik oranın üzerinde."
            )
        return (
            f"{verdict} Yöntem hata payı <{self.estimator_error_pp} puan. "
            "Bu oran 'belgenin şu kadarı makine tarafından yazıldı' demek değildir; "
            "belgede **LLM ile değiştirilmiş** bölüm oranıdır ve insan metni "
            "içerir."
        )

    def to_dict(self, share: float | None = None) -> dict[str, object]:
        return {
            "arxiv_cs_abstracts": self.arxiv_cs_abstracts,
            "arxiv_cs_introductions": self.arxiv_cs_introductions,
            "arxiv_mathematics_abstracts": self.arxiv_math_abstracts,
            "arxiv_mathematics_introductions": self.arxiv_math_introductions,
            "arxiv_eess_abstracts": self.arxiv_eess_abstracts,
            "nature_portfolio_abstracts": self.nature_portfolio_abstracts,
            "pre_chatgpt_baseline": self.pre_chatgpt_baseline,
            "pubmed_2024_lower_bound": self.pubmed_2024_lower_bound,
            "estimator_error_percentage_points": self.estimator_error_pp,
            "disclosure_rate": self.disclosure_rate,
            "peer_review_range": list(self.peer_review_range),
            "least_affected_sections": list(self.least_affected_sections),
            "sample_size": self.sample_size,
            "source": self.source,
            "corroborating_source": self.corroborating_source,
            "caveat": self.caveat,
            "reading": self.reading(share) if share is not None else "",
        }


#: Kobak vd.'s excess-vocabulary set, from >15M PubMed abstracts. These are
#: measured LLM-associated words, not a style opinion, and they work offline for
#: free - which matters because Turkish has no equivalent published list.
LLM_WORD_SIGNALS: dict[str, float] = {
    # Rare-set words, with the measured single-word effect on prevalence.
    "potential": 5.2,
    "underscore": 0.0,
    "intricate": 0.0,
    "meticulous": 0.0,
    "delve": 0.0,
    "pivotal": 0.0,
    "realm": 0.0,
    "comprehensive": 0.0,
    "crucial": 0.0,
    "notably": 0.0,
    "particularly": 0.0,
    "enhancing": 0.0,
    "exhibited": 0.0,
    "insights": 0.0,
    "across": 0.0,
    "additionally": 0.0,
    "within": 0.0,
}

#: The inverse signal, and the more useful half. Thelwall & Kousha (2026) measured
#: across 1.25M science-wide full texts that **LLMs actively avoid ``thus`` and
#: ``moreover``**. Both are high-frequency academic connectives in human prose,
#: so their *absence* relative to a human baseline is positive evidence - and it
#: is evidence a lexicon can compute without a model.
LLM_AVOIDED_CONNECTIVES: frozenset[str] = frozenset(
    {"thus", "moreover", "furthermore", "henceforth"}
)

LLM_WORD_SOURCE = (
    "Kobak vd., Science Advances 2025 (DOI 10.1126/sciadv.adt3813) "
    "excess-vocabulary listesi; Thelwall & Kousha, arXiv:2604.07565 (2026) "
    "1.25M tam metin üzerinde 'thus' ve 'moreover' bastırması."
)


@dataclass(frozen=True, slots=True)
class EnglishStyleReference:
    """Reference values for human English academic prose, by section.

    **Original measurement**, not literature values: 4,462 pre-ChatGPT arXiv
    abstracts (2015-2019, 742,788 tokens) for the document-level figures, and 48
    PMC OA full texts (113,928 tokens of section text) for the per-section ones.
    Pre-ChatGPT is the point: these are unambiguously human-written English
    academic prose, which is exactly the population a false positive is spent.

    Why it matters, in three numbers:

    * **Results prose is repetitive by design.** MTLD 54.6 in Results against
      85.2 in Discussion and 90.7 in Introduction. A low-diversity flag on a
      Results section is weak evidence of anything, and independently agrees with
      Liang vd.'s finding that methods and experiments are the *least*
      LLM-modified sections.
    * **Methods has the shortest sentences** (median 20.9 words) and a
      distinctive function-word profile: ``was``/``were``/``at``/``by`` are all
      elevated relative to every other section. An English stylometric model that
      does not treat Methods separately will not work.
    * **Mathematics is a different language.** MTLD 60.9 against 94-102
      elsewhere, with the highest sentence-length variance. Any threshold keyed
      to English academic prose mis-flags mathematics unless it is
      field-conditioned - and mathematics is also the discipline with the lowest
      measured LLM prevalence (4.1% of introductions).

    MTLD and HD-D are used rather than raw TTR because they are stable above
    ~100 tokens (Koizumi & In'nami, *System* 40(4), 2012); TTR is window-dependent
    and the 0.608 figure below is a fixed 200-token window.

    The L2 row matters more than any other line here. Human L2 writers at CEFR
    A2 measure MTLD ~58 (SD 20.5) against an L1 academic median of 90.6
    (Shatz vd., 2026). **A Turkish student writing an English thesis looks
    lexically poor by L1 academic standards while being entirely human**, and
    this table says so in the row a reader will actually compare against.
    """

    sentence_length_mean: float = 24.87
    sentence_length_sd: float = 5.62
    mtld_mean: float = 93.57
    mtld_sd: float = 31.87
    hdd_mean: float = 0.622
    lexical_density_percent: float = 64.18
    lexical_density_sd: float = 4.45
    ttr_200_window: float = 0.608

    by_discipline_mtld: Mapping[str, float] = field(
        default_factory=lambda: {
            "mathematics": 60.89,
            "economics": 94.19,
            "physics": 94.33,
            "computer_science": 98.43,
            "statistics_ml": 101.56,
            "quantitative_biology": 98.92,
        }
    )
    by_discipline_sentence_length: Mapping[str, float] = field(
        default_factory=lambda: {
            "mathematics": 26.77,
            "physics": 25.62,
            "quantitative_biology": 25.37,
            "computer_science": 24.36,
            "economics": 23.50,
            "statistics_ml": 24.06,
        }
    )

    section_sentence_length_median: Mapping[str, float] = field(
        default_factory=lambda: {
            "introduction": 27.34,
            "methods": 20.91,
            "results": 22.03,
            "discussion": 25.20,
            "conclusion": 24.80,
        }
    )
    section_mtld_median: Mapping[str, float] = field(
        default_factory=lambda: {
            "introduction": 90.7,
            "methods": 81.0,
            "results": 54.6,
            "discussion": 85.2,
            "conclusion": 84.6,
        }
    )

    l2_a2_mtld: float = 58.49
    l2_a2_mtld_sd: float = 22.68
    l2_a1_mtld: float = 45.53

    sample_size: int = 4_462
    section_sample_size: int = 48
    source: str = PROVENANCE["original"]
    mtld_source: str = (
        "Koizumi & In'nami, System 40(4):554-564 (2012); "
        "L2 MTLD: Shatz vd. (2026), Univ. of Birmingham"
    )

    def section_reading(self, role: str) -> str:
        """What a section's own numbers mean, relative to human prose."""
        length = self.section_sentence_length_median.get(role)
        mtld = self.section_mtld_median.get(role)
        if length is None or mtld is None:
            return ""
        notes: list[str] = []
        if role == "results":
            notes.append(
                f"MTLD {mtld} — insan metinde de düşük (tartışma {self.section_mtld_median['discussion']}). "
                "Sonuç bölümüinde düşük çeşitlilik zayıf kanıttır."
            )
        elif role == "methods":
            notes.append(
                f"cümle medyanı {length} kelime — bölümlerin en kısası; yöntem "
                "kurgusunun doğal sonucu, yazım göstergesi değil."
            )
        if role in {"introduction", "conclusion"}:
            notes.append(
                "ölçülmüş olarak en çok LLM değişikliğinin görüldüğü bölümlerden"
            )
        return f"İnsan İngilizce akademik metinde {role}: " + " ".join(notes)

    def l2_note(self) -> str:
        return (
            f"İnsan, ana dilini İngilizce olmayan yazar: A2 seviyesinde MTLD "
            f"{self.l2_a2_mtld} ± {self.l2_a2_mtld_sd} (A1: {self.l2_a1_mtld}) — L1 "
            f"akademik medyanı {self.mtld_mean:.0f}'in altında. Türkçe öğrencinin "
            "İngilizce tezi L1 akademik ölçüte göre leksik olarak fakir görünür ve "
            "tamamen insan yazmış olabilir; bu satır karşılaştırma için burada."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "document_level": {
                "sentence_length_mean": self.sentence_length_mean,
                "sentence_length_sd": self.sentence_length_sd,
                "mtld_mean": self.mtld_mean,
                "mtld_sd": self.mtld_sd,
                "hdd_mean": self.hdd_mean,
                "lexical_density_percent": self.lexical_density_percent,
                "lexical_density_sd": self.lexical_density_sd,
                "ttr_200_token_window": self.ttr_200_window,
            },
            "by_discipline": {
                name: {
                    "mtld": self.by_discipline_mtld.get(name),
                    "sentence_length": self.by_discipline_sentence_length.get(name),
                }
                for name in sorted(self.by_discipline_mtld)
            },
            "by_section": {
                name: {
                    "sentence_length_median": self.section_sentence_length_median.get(name),
                    "mtld_median": self.section_mtld_median.get(name),
                    "reading": self.section_reading(name),
                }
                for name in ("introduction", "methods", "results", "discussion", "conclusion")
            },
            "l2_reference": {
                "a1_mtld": self.l2_a1_mtld,
                "a2_mtld": self.l2_a2_mtld,
                "a2_mtld_sd": self.l2_a2_mtld_sd,
                "note": self.l2_note(),
            },
            "llm_word_signals": dict(LLM_WORD_SIGNALS),
            "llm_avoided_connectives": sorted(LLM_AVOIDED_CONNECTIVES),
            "llm_word_source": LLM_WORD_SOURCE,
            "sample_size": self.sample_size,
            "section_sample_size": self.section_sample_size,
            "provenance": PROVENANCE,
            "source": self.source,
            "mtld_source": self.mtld_source,
            "caveat": (
                "Bu bir araştırma turunun **kendi ölçümüdür**, yayımlanmış bir "
                "tablo değildir. Pre-ChatGPT arXiv özetleri ve PMC tam metinleri; "
                "tez dili değildir ve İngilizce L1 ağırlıklıdır."
            ),
        }