"""Turnitin-style similarity metrics, with the filters Turkish theses use.

Universities do not read "you have 3 matches". They read a percentage that was
produced with a specific filter set, and they compare it to a number written
into their own regulation. The filters that appear in the Turkish institutional
rules (YÖK sets **no** national percentage; each institute board decides) are:

* ``Kaynakça hariç`` - reference list excluded
* ``Alıntılar dahil`` - quotations included in the headline number, and a second
  number with quotations excluded
* ``5 kelimeden az eşleşmeler hariç`` - matches shorter than ~5 words excluded
* a per-source ceiling, because a single source is the usual actual problem

Observed institutional thresholds (see :data:`INSTITUTION_BANDS` for sources):
15% excluding quotations / 20% including them / 2% single source at Yıldız
Technical University; 20% at İstanbul and Başkent; 30% at Eskişehir, Çukurova and
Akdeniz.

Because these numbers describe *coverage against a source set we were given*, this
module also prints the measured Turkish baseline it should be read against:
28.7% mean similarity (SD 11.58, n=600 education-sciences theses; 29.31% for
Turkish-language and 24.37% for English-language theses, Toprak 2014). A 30%
overlap is unremarkable for a Turkish thesis and alarming for an English one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from checker_app.domain.enums import CitationStatus
from checker_app.domain.models import MatchSpan
from checker_app.domain.text import Token
from checker_app.services.attribution import Span

__all__ = [
    "InstitutionBand",
    "INSTITUTION_BANDS",
    "SourceShare",
    "SimilarityStats",
    "SimilarityReport",
    "TurkishBaseline",
    "TurkishAiPresence",
    "build_similarity_report",
]


@dataclass(frozen=True, slots=True)
class InstitutionBand:
    """A published institutional rule."""

    institution: str
    total_excl_quotes: float | None
    total_incl_quotes: float | None
    single_source: float | None
    url: str
    note: str = ""


# Percentages, exactly as published by each institution (2023-2026 access dates).
INSTITUTION_BANDS: tuple[InstitutionBand, ...] = (
    InstitutionBand(
        "YTÜ (Temiz Enerji / Sosyal Bilimler / Fen Bilimleri)",
        15.0,
        20.0,
        2.0,
        "https://tet.yildiz.edu.tr/mezuniyet/intihal-kontrol",
        "iki rapor (savunma öncesi/sonrası), uyumlu olmayan tez savunmaya giremez",
    ),
    InstitutionBand(
        "İstanbul Üniversitesi Sağlık Bilimleri",
        None,
        20.0,
        None,
        "http://cdn.istanbul.edu.tr/statics/saglikbilimleri.istanbul.edu.tr/wp-content/uploads/2018/02/Alper-Hoca-2.pdf",
        "bibliyograf hariç, alıntılar dahil, 5 kelimeden küçük eşleşmeler hariç",
    ),
    InstitutionBand(
        "Başkent Üniversitesi Enstitüler",
        20.0,
        20.0,
        2.0,
        "https://www.baskent.edu.tr/belgeler/mevzuat/yonerge/enstitu_yong_16.pdf",
        "tek kaynak sınırı %2",
    ),
    InstitutionBand(
        "Eskişehir Teknik Üniversitesi LEE",
        30.0,
        30.0,
        15.0,
        "https://lee.eskisehir.edu.tr/tr/Duyuru/Detay/tezlerin-intihal-kontrolunden-gecirilmesi-hakkinda-bilgilendirme",
        "izin verilen filtreler: kaynakça hariç, %1 altı örtüşme hariç",
    ),
    InstitutionBand(
        "Çukurova Üniversitesi BADI / Akdeniz Üniversitesi SBE",
        30.0,
        30.0,
        None,
        "https://babe.cu.edu.tr/cu/ogrenci/turnitin-intihal-benzesim-programi-kullanim-ilkeleri-ve-kullanim-kilavuzu-turnitin-plagiarism-program/turnitin-intihal-benzesim-programi-kullanim-ilkeleri",
        "%30 üzeri yazılı açıklama ister; kurum bunun 'hukuken intihal yok' demek olmadığını vurgular",
    ),
)


@dataclass(frozen=True, slots=True)
class TurkishBaseline:
    """The measured distribution this percentage should be read against."""

    mean_percent: float = 28.7
    sd_percent: float = 11.58
    turkish_percent: float = 29.31
    english_percent: float = 24.37
    high_plagiarism_percent: float = 34.5
    sample_size: int = 600
    field: str = "eğitim bilimleri"
    source: str = (
        "Ziya Toprak, Türkiye'de Akademik Yazı: İntihal ve Özgünlük, "
        "Boğaziçi Üniversitesi Eğitim Dergisi 34(2), 2014"
    )
    caveat: str = (
        "Zaman ve alan dar; disiplinsel kırılım için 2025-26 döneminde doğrulanmış "
        "Türkçe tez oranı yayımlanmamıştır."
    )

    def percentile_note(self, percent: float) -> str:
        if self.mean_percent <= 0:
            return ""
        sigma = (percent - self.mean_percent) / self.sd_percent
        return f"ölçülen ortalama {self.mean_percent}% ± {self.sd_percent} → z={sigma:+.2f}"


@dataclass(frozen=True, slots=True)
class TurkishAiPresence:
    """What Turkish academic writing actually looks like, measured.

    This is the only published distribution for Turkish academic prose that the
    research turned up, and it is about AI presence rather than similarity -
    so it is the right companion to :class:`TurkishBaseline`, which measures
    overlap. Akkaya & Beygirci (2026, DOI 10.46452/baksoder.1899625) hand-verified
    **204 Turkish articles** from 68 university journals on DergiPark, published
    2025:

    ==================  =========  ==========
    AI-rate band         articles   band mean
    ==================  =========  ==========
    0-20%                    122  (59.8%)   6%
    21-40%                    45  (22.1%)  28%
    41-60%                    25  (12.3%)  50%
    61-80%                    11   (5.4%)  69%
    81-100%                    1   (0.5%)  94%
    **total**                204          mean **20%**
    ==================  =========  ==========

    Control groups from the same study: 20 AI-generated documents were **100%**
    flagged at "generally 40-60%", and 2010-2020 Turkish articles came out
    **0-10%**. Indexed in TR Dizin (n=102) the mean was **12%**, against **28%**
    for non-TR Dizin (n=102); articles above 40% were 9/102 (8.8%) against
    28/102 (27.5%).

    Two readings matter for a thesis:

    * **59.8% of Turkish academic articles sit below a 20% AI rate**, with a band
      mean of 6% - which is the same 20% this tool uses as its reporting floor,
      arrived at from the opposite direction.
    * **AI concentrates in the opening, not the method.** By section: introduction
      and literature review 100/204 (**49.0%**), interpretation of findings 94
      (46.1%), abstract 78 (38.2%), conclusion 77 (37.7%), **method 6 (2.9%)**.
      A thesis whose method section is fine and whose literature review is not
      is the ordinary case, not the suspicious one.

    Caveat carried with it: articles, not theses, and a hand-verified sample of
    one journal platform.
    """

    mean_percent: float = 20.0
    below_20_percent: int = 122
    below_20_share: float = 0.598
    below_20_band_mean_percent: float = 6.0
    tr_dizin_mean_percent: float = 12.0
    non_tr_dizin_mean_percent: float = 28.0
    sample_size: int = 204
    section_rates: Mapping[str, float] = field(
        default_factory=lambda: {
            "introduction_literature": 0.490,
            "interpretation_of_findings": 0.461,
            "abstract": 0.382,
            "conclusion": 0.377,
            "method": 0.029,
        }
    )
    source: str = (
        "Akkaya & Beygirci (2026), MAKALELERDE YAPAY ZEKÂ VARLIĞININ "
        "DEĞERLENDİRİLMESİ, DOI 10.46452/baksoder.1899625"
    )
    caveat: str = (
        "Makale, tez değil; DergiPark'taki 68 üniversite dergisinden elle "
        "doğrulanmış 204 örnek. Tezler için ölçülmüş bir AI oranı dağılımı yok."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_percent": self.mean_percent,
            "below_20_percent_count": self.below_20_percent,
            "below_20_percent_share": self.below_20_share,
            "below_20_percent_band_mean": self.below_20_band_mean_percent,
            "tr_dizin_mean_percent": self.tr_dizin_mean_percent,
            "non_tr_dizin_mean_percent": self.non_tr_dizin_mean_percent,
            "sample_size": self.sample_size,
            "section_rates": dict(self.section_rates),
            "source": self.source,
            "caveat": self.caveat,
            "reading": (
                f"Türkçe akademik yazıda ortalama AI oranı %{self.mean_percent}; "
                f"{self.below_20_percent}/{self.sample_size} (%{self.below_20_share * 100:.1f}) "
                f"makale %20'nin altında ve o bandın ortalaması yalnız "
                f"%{self.below_20_band_mean_percent}. Bu araç %20'yi raporlama "
                "tabanı olarak kullanıyor; Türkçe veriden bağımsız olarak aynı "
                "eşik çıkıyor. AI giriş ve literatür taramasında yoğunlaşıyor "
                f"(%{self.section_rates['introduction_literature'] * 100:.1f}), "
                f"yöntemde neredeyse hiç yok (%{self.section_rates['method'] * 100:.1f})."
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceShare:
    """How much of the document one reference accounts for."""

    source_id: str
    name: str
    matched_words: int
    share_percent: float
    match_count: int
    quoted_words: int = 0
    unattributed_words: int = 0
    uncited_matches: int = 0

    @property
    def attributed(self) -> bool:
        return self.uncited_matches == 0


@dataclass(frozen=True, slots=True)
class SimilarityStats:
    """The numbers behind the percentages, all recomputable."""

    total_words: int
    matched_words: int
    matched_words_quoted: int
    matched_words_in_references: int
    reference_words: int
    quoted_words: int
    match_count: int
    cases: int
    granularity: float
    """Matches per connected case. The PAN metric punishes fragmentation, and
    so should this report: one copied paragraph reported five times looks worse
    than the same paragraph reported once."""


@dataclass(frozen=True, slots=True)
class SimilarityReport:
    """Everything an academic-integrity report needs, plus its provenance."""

    similarity_incl_quotes: float
    similarity_excl_quotes: float
    similarity_excl_references: float
    similarity_excl_references_quotes: float
    stats: SimilarityStats
    per_source: tuple[SourceShare, ...] = ()
    largest_single_source: SourceShare | None = None
    attribution: dict[str, int] = field(default_factory=dict)
    bands: tuple[dict[str, object], ...] = ()
    baseline: TurkishBaseline = field(default_factory=TurkishBaseline)
    ai_presence: TurkishAiPresence = field(default_factory=TurkishAiPresence)
    """What AI presence in Turkish academic writing actually looks like, so the
    AI share has a Turkish reference the way the similarity percentage does."""
    filters: str = "kaynakça hariç · alıntılar dahil · 5 kelimeden az eşleşmeler hariç"
    filters_applied: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "similarity": {
                "incl_quotes_percent": round(self.similarity_incl_quotes, 2),
                "excl_quotes_percent": round(self.similarity_excl_quotes, 2),
                "excl_references_percent": round(self.similarity_excl_references, 2),
                "excl_references_and_quotes_percent": round(
                    self.similarity_excl_references_quotes, 2
                ),
                "largest_single_source_percent": (
                    round(self.largest_single_source.share_percent, 2)
                    if self.largest_single_source
                    else 0.0
                ),
                "largest_single_source_name": (
                    self.largest_single_source.name if self.largest_single_source else None
                ),
            },
            "filters": {
                "description": self.filters,
                "applied": self.filters_applied,
            },
            "stats": {
                "total_words": self.stats.total_words,
                "matched_words": self.stats.matched_words,
                "matched_words_quoted": self.stats.matched_words_quoted,
                "matched_words_in_references": self.stats.matched_words_in_references,
                "reference_words": self.stats.reference_words,
                "quoted_words": self.stats.quoted_words,
                "match_count": self.stats.match_count,
                "cases": self.stats.cases,
                "granularity": round(self.stats.granularity, 3),
            },
            "attribution": self.attribution,
            "per_source": [
                {
                    "source_id": s.source_id,
                    "name": s.name,
                    "matched_words": s.matched_words,
                    "share_percent": round(s.share_percent, 2),
                    "match_count": s.match_count,
                    "quoted_words": s.quoted_words,
                    "unattributed_words": s.unattributed_words,
                    "uncited_matches": s.uncited_matches,
                }
                for s in self.per_source
            ],
            "institutional_bands": list(self.bands),
            "baseline": {
                "mean_percent": self.baseline.mean_percent,
                "sd_percent": self.baseline.sd_percent,
                "turkish_percent": self.baseline.turkish_percent,
                "english_percent": self.baseline.english_percent,
                "high_plagiarism_percent": self.baseline.high_plagiarism_percent,
                "sample_size": self.baseline.sample_size,
                "field": self.baseline.field,
                "source": self.baseline.source,
                "caveat": self.baseline.caveat,
                "note": self.baseline.percentile_note(self.similarity_incl_quotes),
            },
            "turkish_ai_presence": self.ai_presence.to_dict(),
        }


def build_similarity_report(
    *,
    tokens: Sequence[Token],
    matches: Sequence[MatchSpan],
    total_words: int,
    reference_token_indices: frozenset[int] | set[int],
    quote_spans: Sequence[Span] = (),
    statuses: dict[int, CitationStatus] | None = None,
    min_match_words: int = 5,
    exclude_references: bool = True,
    include_quotes: bool = True,
) -> SimilarityReport:
    """Compute the institutional-style percentages.

    ``statuses`` maps ``id(match)`` -> :class:`CitationStatus`; the attribution
    split is what keeps a correctly quoted passage from counting as misconduct.
    """
    statuses = statuses or {}
    starts = [token.char_start for token in tokens]
    covered: set[int] = set()
    per_source_covered: dict[str, set[int]] = {}
    match_count_by_source: dict[str, int] = {}
    uncited_by_source: dict[str, int] = {}
    quoted_by_source: dict[str, set[int]] = {}
    unattributed_by_source: dict[str, set[int]] = {}
    reference_indices = set(reference_token_indices) if exclude_references else set()

    # Computed either way: with include_quotes=False the quoted words still have
    # to leave the numbers, otherwise the flag does nothing.
    quoted_indices = _indices_in_spans(starts, quote_spans)
    counted_quotes = quoted_indices if include_quotes else set()
    kept: list[MatchSpan] = []
    for match in matches:
        if match.matched_words < min_match_words:
            continue
        indices = _indices_for_span(starts, match.doc_char_start, match.doc_char_end)
        if not indices:
            continue
        if reference_indices and indices <= reference_indices:
            # A match that lives entirely inside the reference list is an
            # artefact of comparing against sources, not a finding.
            continue
        kept.append(match)
        covered |= indices
        per_source_covered.setdefault(match.source_id, set()).update(indices)
        match_count_by_source[match.source_id] = match_count_by_source.get(match.source_id, 0) + 1
        status = statuses.get(id(match))
        if status is CitationStatus.NOT_CITED_OR_QUOTED:
            uncited_by_source[match.source_id] = uncited_by_source.get(match.source_id, 0) + 1
        quote_hits = indices & counted_quotes
        if quote_hits:
            quoted_by_source.setdefault(match.source_id, set()).update(quote_hits)
        if status is not CitationStatus.CITED_AND_QUOTED:
            unattributed_by_source.setdefault(match.source_id, set()).update(
                indices - quote_hits if include_quotes else indices
            )

    denominator = max(1, total_words)
    excl_ref = covered - reference_indices
    excl_ref_quotes = excl_ref - quoted_indices
    matched_quotes = covered & quoted_indices
    # With --excl-quotes the quoted words leave the report entirely, so both
    # headline numbers are the same by construction.
    headline = covered if include_quotes else excl_ref_quotes

    cases = _count_cases(kept)
    stats = SimilarityStats(
        total_words=total_words,
        matched_words=len(covered),
        matched_words_quoted=len(matched_quotes),
        matched_words_in_references=len(covered & reference_indices),
        reference_words=len(reference_indices),
        quoted_words=len(quoted_indices),
        match_count=len(kept),
        cases=cases,
        granularity=round(len(kept) / cases, 3) if cases else 0.0,
    )

    per_source = tuple(
        SourceShare(
            source_id=source_id,
            name=next(m.source_name for m in kept if m.source_id == source_id),
            matched_words=len(indices),
            share_percent=100.0 * len(indices) / denominator,
            match_count=match_count_by_source.get(source_id, 0),
            quoted_words=len(quoted_by_source.get(source_id, set())),
            unattributed_words=len(unattributed_by_source.get(source_id, set())),
            uncited_matches=uncited_by_source.get(source_id, 0),
        )
        for source_id, indices in sorted(per_source_covered.items(), key=lambda kv: -len(kv[1]))
    )
    largest = per_source[0] if per_source else None

    report = SimilarityReport(
        similarity_incl_quotes=100.0 * len(headline) / denominator,
        similarity_excl_quotes=100.0 * len(excl_ref_quotes) / denominator,
        similarity_excl_references=100.0 * len(excl_ref) / denominator,
        similarity_excl_references_quotes=100.0 * len(excl_ref_quotes) / denominator,
        stats=stats,
        per_source=per_source,
        largest_single_source=largest,
        attribution={
            status.value: sum(1 for m in kept if statuses.get(id(m)) is status)
            for status in CitationStatus
        },
        filters_applied={
            "exclude_references": exclude_references,
            "include_quotes": include_quotes,
            "min_match_words": min_match_words,
        },
    )
    return replace(report, bands=_evaluate_bands(report))


def _evaluate_bands(report: SimilarityReport) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for band in INSTITUTION_BANDS:
        issues: list[str] = []
        if band.total_excl_quotes is not None and (
            report.similarity_excl_quotes > band.total_excl_quotes
        ):
            issues.append(
                f"alıntılar hariç {report.similarity_excl_quotes:.1f}% > {band.total_excl_quotes:.0f}%"
            )
        if band.total_incl_quotes is not None and (
            report.similarity_incl_quotes > band.total_incl_quotes
        ):
            issues.append(
                f"alıntılar dahil {report.similarity_incl_quotes:.1f}% > {band.total_incl_quotes:.0f}%"
            )
        largest = report.largest_single_source
        if (
            band.single_source is not None
            and largest
            and (largest.share_percent > band.single_source)
        ):
            issues.append(
                f"tek kaynak {largest.share_percent:.1f}% > {band.single_source:.0f}%"
                f" ({largest.name})"
            )
        rows.append(
            {
                "institution": band.institution,
                "url": band.url,
                "note": band.note,
                "thresholds": {
                    "total_excl_quotes_percent": band.total_excl_quotes,
                    "total_incl_quotes_percent": band.total_incl_quotes,
                    "single_source_percent": band.single_source,
                },
                "issues": issues,
                "status": "uygun" if not issues else "gözden geçirilmeli",
            }
        )
    return tuple(rows)


def _indices_for_span(starts: Sequence[int], char_start: int, char_end: int) -> set[int]:
    from bisect import bisect_left, bisect_right

    first = bisect_left(starts, char_start)
    last = bisect_right(starts, char_end - 1)
    return set(range(first, max(first, last)))


def _indices_in_spans(starts: Sequence[int], spans: Sequence[Span]) -> set[int]:
    out: set[int] = set()
    for span in spans:
        out |= _indices_for_span(starts, span.char_start, span.char_end)
    return out


def _count_cases(matches: Sequence[MatchSpan]) -> int:
    """Connected clusters of matches: one copied passage = one case.

    Clusters are cut on the *document* side only. The PAN granularity penalty
    exists to punish fragmenting one true case into many detections, and a
    passage found in three sources is still one case - counting it three times
    would overstate the fragmentation it is meant to measure.
    """
    if not matches:
        return 0
    ordered = sorted(matches, key=lambda m: m.doc_char_start)
    cases = 0
    current_end = None
    for span in ordered:
        if current_end is None or span.doc_char_start > current_end:
            cases += 1
        current_end = max(current_end or 0, span.doc_char_end)
    return max(1, cases)
