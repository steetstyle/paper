"""Reference-list audit: structural plausibility of every cited source.

Why this module exists
----------------------
Fabricated references are the best-evidenced research-integrity problem of the
moment, and they are exactly what a thesis committee can act on. Topaz et al.,
*The Lancet* (2026, DOI 10.1016/S0140-6736(26)00603-3) audited 2.5M biomedical
papers / 126M structured references from Jan 2023 to Feb 2026 and found **4,046
fabricated references in 2,810 papers** out of 97.1M verified ones. The rate rose
**more than twelvefold in three years**:

| period | rate | per 10,000 papers |
|---|---|---|
| 2023 | 1 in 2,828 | 4.0 |
| 2025 | 1 in 458 | 51.3 |
| first 7 weeks of 2026 | 1 in 277 | 56.9 |

Other measured anchors from the same literature:

* **91%** detector precision on that corpus, and **91%** of affected papers carry
  only **1-2** fabricated references - so a small number of bad entries is the
  normal case, not an outlier, and "only two look odd" is not reassurance.
* In CS venues: ~300 papers with at least one hallucinated citation across
  ACL/NAACL/EMNLP 2024-2025 (arXiv:2601.18724); roughly **1 in 20** NeurIPS and
  USENIX Security 2025 papers has **at least two** (arXiv:2607.00738); and HPC
  conference proceedings went from **0%** affected in 2021 to **2-6%** in every
  2025 proceedings, with no author disclosing AI use despite all four policies
  requiring it (arXiv:2602.05867).
* 2025 CS papers: **2.6%** carried at least one potentially hallucinated
  citation, up from **~0.3%** in 2024 (*Nature* 652, 26-29, 2026,
  DOI 10.1038/d41586-026-00969-z).
* The published prior is that **30-69%** of LLM-generated biomedical references
  are fabricated, and applying a plagiarism rubric to 50 LLM-generated research
  proposals found **24.0%** plagiarized (36.0% counting unverifiable claims),
  against **0.8-6.3%** for human-written papers on PeerRead (arXiv:2502.16487).
* Commercial tools do not help. In that same study **Turnitin caught 0%** and
  **OpenScholar caught 0%** of the verified cases.

Adelphi University's 2025 guidance for faculty lists exactly this among the
corroborating features that make an AI-misconduct report actionable - alongside
argument progression matching an AI output - and requires that a detector score
alone is "grounds for further investigation, **not** sufficient evidence". That
is the role this module plays: it supplies corroboration a committee can check
by hand, offline, in seconds.

What this module can and cannot do
----------------------------------
It is **structural only**. It cannot resolve a DOI, query CrossRef, or read a PDF.
Every check is "does this entry look like something a person could look up?",
never "is this entry real?". With no network there is no way to establish
existence, and the honest failure mode of a resolver is a *false accusation* -
PAN-2025 measured embedding baselines flagging genuine paraphrased text twice as
often as actual plagiarism (arXiv:2510.06805).

Two limits stated plainly:

* **No measured thesis-specific rate exists.** Every number above is for journal
  articles and conference papers. This is a genuine gap in the literature.
* **Absence of an identifier is not fabrication.** Plenty of legitimate Turkish
  and humanities references are print-only with no DOI, so ``no_identifier``
  alone is ``note``, not ``required``. Only a cluster of problems, or a
  malformed identifier, escalates.

Three more limits, each from a measurement rather than from caution
-------------------------------------------------------------------
* **Positive predictive value, not F1.** HALLMARK's base-rate sweep
  (arXiv:2607.18360) shows that at a venue-realistic 1-2% hallucination rate the
  best verifier still yields only **5-18% PPV** - four to nine false alarms per
  true catch (Opus 4.7 9.5% at 1%, Sonnet 4.6 5.8%, GPT-5.1 2.0%). This module
  reports the flag count with that frame attached, so a reader does not read
  "3 flags" as "3 problems".
* **Consensus, not any-no-match.** The highest-value and cheapest finding in that
  same paper: holding the databases and matcher fixed, flagging when *any* source
  fails to confirm yields FPR **0.729**, while flagging only when *every* source
  fails yields **0.049** - a ~15x cut with no extra data and zero cost, at the
  price of recall dropping from 0.835 to 0.251. Adding positive-contradiction
  checks recovers it to DR 0.865 / FPR 0.092 / MCC 0.771, beating every LLM on
  MCC while making no LLM call. That is why "not found" is never a standalone
  reason here, and why escalation needs a cluster.
* **Extraction beats matching.** RefChecker's measured FPR is **50.7%** - 36 of 71
  genuinely real references flagged - and Phantom References states plainly that
  most flags were *not* hallucinations but extraction artifacts: mangled author
  strings, truncated titles, an editor counted as co-author, a title fragment
  read as an author, two real papers with the same title. This module's own
  extraction is crude (sentence splitting plus paragraph re-joining), so its flags
  are a list to look up by hand, not an automatic judgement.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from checker_app.domain.models import Sentence
from checker_app.domain.sections import SectionRole

__all__ = ["ReferenceAudit", "ReferenceEntry", "audit_references"]

# arXiv: YYMM.NNNNN (4 or 5 digits since 2015-01) or the pre-2007 archive form.
_ARXIV_RE = re.compile(r"arxiv[:\s/]*(\d{2})(0[1-9]|1[0-2])[.\-]?(\d{4,5})", re.I)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9<>+]+", re.I)
_DOI_LOOSE_RE = re.compile(r"\bdoi[:\s]*", re.I)
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

#: A four-digit number followed by a comma sits in the author/venue gap, which is
#: where the publication year goes. Page numbers are followed by spaces or by
#: further back-references, not by a comma.
_AUTHOR_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*,")
#: The floor for "this year cannot be real".
#:
#: This used to be 1950, on the assumption that a thesis cites only modern
#: literature. Measurement on a real thesis destroyed that assumption: of the nine
#: entries it flagged as impossible-old-year, every one was legitimate - Hubble
#: 1926, Zwicky 1933 and 1937, Smith 1936 - because physics and astronomy cite
#: foundational work routinely. Flagging those is not a weak signal, it is a wrong
#: one, and it is the kind that costs a reader trust in every other flag.
#:
#: So the flag now fires only below the first scientific periodicals (the
#: Philosophical Transactions, 1665), where a *cited reference* really is
#: implausible. Pre-1950 remains accepted, because it has to be.
MIN_PLAUSIBLE_YEAR = 1665
# A surname-initial pair: "Yılmaz, A." / "Smith J." / "Kumar, S. & Patel, R."
_INITIAL_RE = re.compile(r"(?:^|[\s,;.])([A-ZÇĞİÖŞÜ])[.\s]?(?=[,.;\s]|$)")

#: Venues a real thesis reference must name. Absence of *something* in this slot
#: is the single most common shape of an invented entry.
_VENUE_HINT_RE = re.compile(
    r"\b(?:dergi|journal|review|proceedings|proc\.|conf\.|kitap|book|"
    r"yayın|publish|press|university|universit|üniversit|"
    r"volume|vol\.|cilt|issue|no\.|sayı|pp?\.|sayfa|"
    r"türk|journal|thesis|tez|sempoz|conference|konferans|"
    r"doi|arxiv|isbn|ssn|ieee|acm|springer|elsevier|wiley|"
    # No trailing \b: Turkish suffixes would defeat it ("Dergisi" never ends on a
    # word boundary after "dergi"), and a match on a word prefix is harmless here.
    r"taylor|francis|sage|mdpi|plos|nature|science|pubmed|scopus)",
    re.I,
)
_SENTENCE_LIKE_RE = re.compile(r"\b(?:neden|sonuç|bulgu|amaç|yöntem|öneri)\b", re.I)


#: HALLMARK (arXiv:2607.18360) sweeps PPV at realistic base rates. At 1-2%
#: prevalence the best verifiers manage 5-18%, i.e. four to nine false alarms per
#: true catch. Academic theses are not a measured population (see the module
#: docstring), so the frame is borrowed rather than claimed.
POSITIVE_PREDICTIVE_VALUE: dict[str, object] = {
    # Measured hallucination rate for academic-paper-shaped references across
    # ICLR/ICML/NeurIPS/USENIX Security 2025: 0.31% to 0.81%
    # (Phantom References, arXiv:2607.00738).
    "academic_paper_base_rate": (0.0031, 0.0081),
    # PPV of the strongest verifiers at 1-2% prevalence (HALLMARK,
    # arXiv:2607.18360).
    "verified_pp_range": (0.02, 0.18),
    "false_alarms_per_true_catch": "4-9",
    # 36 of 71 genuinely real references flagged; most flags were extraction
    # artifacts, not fabrications (Phantom References).
    "refchecker_measured_fpr": 0.507,
}


def expected_true_findings(flagged: int) -> str:
    """How many of ``flagged`` flags are plausibly real.

    Deliberately returns a range rather than a number: the base rate is not
    measured for theses, so any point estimate would be invented precision. The
    arithmetic uses the 0.31-0.81% rate measured for academic papers and the
    5-18% PPV measured for the strongest verifiers at that prevalence.
    """
    if flagged <= 0:
        return "0"
    low = flagged * 0.0031 * 0.02
    high = flagged * 0.0081 * 0.18
    if high < 1:
        return f"muhtemelen 0 (en fazla ~{high:.1f}); {flagged} işaretin çoğu normal"
    return f"~{low:.1f}–{high:.1f} (yani {flagged} işaretin çoğu normal)"


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """One parsed reference line and what is questionable about it."""

    index: int
    text: str
    year: int | None
    has_identifier: bool
    identifier: str | None
    identifier_kind: str | None
    has_authors: bool
    has_venue: bool
    flags: tuple[str, ...]
    risk: str = "note"
    """``ok`` | ``note`` | ``review``. Never ``required``: structural evidence
    alone cannot establish that a source does not exist."""

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text[:300],
            "year": self.year,
            "identifier": self.identifier,
            "identifier_kind": self.identifier_kind,
            "has_authors": self.has_authors,
            "has_venue": self.has_venue,
            "risk": self.risk,
            "flags": list(self.flags),
        }


@dataclass(frozen=True, slots=True)
class ReferenceAudit:
    """The audit, its counts, and the evidence behind treating it seriously."""

    entries: tuple[ReferenceEntry, ...]
    scanned_section: str = ""
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def flagged(self) -> tuple[ReferenceEntry, ...]:
        return tuple(e for e in self.entries if e.risk != "ok")

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            for flag in entry.flags:
                counts[flag] = counts.get(flag, 0) + 1
        flagged = len(self.flagged)
        return {
            "references_scanned": self.total,
            "needs_review": flagged,
            "with_identifier": sum(1 for e in self.entries if e.has_identifier),
            "without_identifier": sum(1 for e in self.entries if not e.has_identifier),
            "without_year": sum(1 for e in self.entries if e.year is None),
            "risk_counts": {
                level: sum(1 for e in self.entries if e.risk == level)
                for level in ("ok", "note", "review")
            },
            "flag_counts": counts,
            "scanned_section": self.scanned_section,
            "expected_true_findings": expected_true_findings(flagged),
            "positive_predictive_value": POSITIVE_PREDICTIVE_VALUE,
        }

    def reading(self) -> str:
        flagged = len(self.flagged)
        if not flagged:
            return (
                f"{self.total} kayıt tarandı, gözden geçirilecek kayıt yok. "
                "**Bu, kaynakların var olduğunun kanıtı değildir** — DOI "
                "çözülmediği için varlık da doğrulanmamıştır."
            )
        return (
            f"{self.total} kayıttan {flagged} tanesi gözden geçirilmeli. "
            f"Uydurma kaynak oranı ~%0.4–0.7 (konferanslarda ölçülen alt sınır) "
            "olduğundan, bu işaretlerin yaklaşık 1–3'ü gerçek bir soruna, "
            "kalanı normal ya da çıkarım hatasına işaret eder. "
            "RefChecker'ın ölçülen FPR'si %50.7'dir ve çoğu işaret uydurma "
            "kaynak değil, bozuk girdi çıkarımıdır."
        )

    def to_dict(self, limit: int = 40, *, only_flagged: bool = True) -> dict[str, object]:
        payload = self.summary()
        rows = self.flagged if only_flagged else self.entries
        payload["entries"] = [e.to_dict() for e in rows[: max(1, limit)]]
        payload["truncated"] = len(rows) > max(1, limit)
        payload["evidence"] = list(self.evidence)
        payload["reading"] = self.reading()
        payload["positive_predictive_value"] = POSITIVE_PREDICTIVE_VALUE
        payload["expected_true_findings"] = self.summary()["expected_true_findings"]
        payload["caveats"] = [
            "Bu denetim yalnızca YAPISALDIR: DOI çözülmez, CrossRef sorgulanmaz, PDF okunmaz. "
            "Bir kaydın gerçekten var olduğunu kanıtlayamaz.",
            "İşaretlerin çoğu gerçek bir sorun DEĞİLDİR. HALLMARK'ın gerçek dağılım "
            "taramasında %1-2 temel oranında en iyi doğrulayıcı bile yalnız %5-18 "
            "teşhis oranı veriyor (gerçek bulgu başına 4-9 yanlış alarm), ve "
            "RefChecker'ın ölçülen FPR'si %50.7 - çoğu işaret uydurma kaynak değil, "
            "bozuk girdi çıkarımıdır (arXiv:2607.18360, arXiv:2607.00738).",
            "Bu modülün kendi girdi çıkarımı da kaba (cümle bölme + paragraf "
            "birleştirme), yani araştırmanın ölçtüğü aynı sınıf hatadan "
            "kaynaklanıyor. Her işaret elle aranmalıdır.",
            "Tezler için ölçülmüş uydurma kaynak oranı YOKTUR (yukarıdaki tüm "
            "sayılar makale/konferans kayıtlarıdır).",
            "Kanıtlanmış risk kümelenmesi tipiktir: The Lancet 2026 denetiminde etkilenen "
            "makalelerin %91'i yalnızca 1-2 uydurma kaynak içeriyordu.",
        ]
        return payload


def audit_references(
    sentences: Sequence[Sentence],
    *,
    this_year: int | None = None,
) -> ReferenceAudit:
    """Vet the reference list structurally.

    Entries are split on the convention real bibliographies use (one reference
    per numbered paragraph or per ``Author. (Year).`` break), because a thesis
    reference list is a list, not a paragraph.
    """
    current_year = this_year or 2026
    reference_lines = [
        s for s in sentences if s.location.section_role is SectionRole.REFERENCES
    ]
    if not reference_lines:
        return ReferenceAudit(
            entries=(),
            scanned_section="",
            evidence=(),
        )

    section = reference_lines[0].location.section_path[-1] if reference_lines else ""
    raw: list[tuple[str, int]] = []
    for text, sentence_index in _entries_from_sentences(reference_lines):
        for piece in _split_entry(text):
            if piece.strip():
                raw.append((piece.strip(), sentence_index))

    entries = tuple(
        _vet(piece, index, current_year) for index, (piece, _) in enumerate(raw, start=1)
    )
    return ReferenceAudit(
        entries=entries,
        scanned_section=section,
        evidence=(
            "The Lancet 2026 (DOI 10.1016/S0140-6736(26)00603-3): 2,5M makale / 126M "
            "yapılandırılmış kayıt taramasında 4.046 uydurma kaynak, 2.810 makale; oran "
            "3 yılda 12 kat arttı (2023'te 1/2.828 → 2025'te 1/458 → 2026'nın ilk 7 "
            "haftasında 1/277).",
            "arXiv:2601.18724: ACL/NAACL/EMNLP 2024-25'te ~300 makalede en az bir "
            "halüsinasyon kaynak; EMNLP 2025'in yarısı tek başına.",
            "arXiv:2602.05867: HPC konferanslarında 2021'de %0 → 2025'te her kongrede "
            "%2-6; hiçbir yazar AI kullanımı beyan etmemiş.",
            "DOI 10.1038/d41586-026-00969-z: 2025 bilgisayar bilimleri makalelerinin "
            "%2,6'sında en az bir şüpheli kaynak (2024: ~%0,3).",
            "Adelphi Üniversitesi 2025: tek başına dedektör puanı 'daha fazla "
            "inceleme gerekçesi' sayılır, yeterli kanıt değildir; doğrulayıcı kanıt "
            "olarak halüsinasyonlu/eksik kaynaklar sayılır.",
        ),
    )


def _entries_from_sentences(lines: Sequence[Sentence]) -> list[tuple[str, int]]:
    """Re-join sentences into whole bibliography entries.

    A reference is not a sentence, and the segmenter quite correctly splits
    ``Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri.`` at its periods. So
    the audit works from *paragraph* runs instead: consecutive sentences that
    share a paragraph index and are adjacent in the character stream are one
    entry. Splitting on the paragraph boundary is what the bibliography's own
    layout already tells us.
    """
    merged: list[tuple[str, int]] = []
    current = ""
    current_paragraph: int | None = None
    previous_end = -1
    previous_line: int | None = None

    def flush() -> None:
        nonlocal current
        if current:
            merged.append((current, previous_end))
            current = ""

    for sentence in lines:
        if sentence.location.block_type.value == "heading":
            flush()
            current_paragraph = None
            previous_end = -1
            previous_line = None
            continue
        paragraph = sentence.location.paragraph_index
        # A printed bibliography puts one reference per *line*, and a segmenter
        # that has just read PDF text will hand back a "sentence" spanning several
        # of them, newline-separated. Measured on a 154-page astrophysics thesis
        # (arXiv:1407.6566), paragraph boundaries alone found 9 entries where the
        # bibliography holds hundreds: every entry ran into the next because PDF
        # text carries no blank lines. So the newlines inside the sentence are
        # followed first, and a line starts a new entry only once what has
        # accumulated already reads as finished - which keeps a reference that
        # wraps across two lines in one piece.
        pieces = sentence.text.split("\n") if "\n" in sentence.text else [sentence.text]
        for offset, piece in enumerate(pieces):
            line_start = sentence.location.line_start + offset
            if offset > 0 and _looks_complete(current):
                flush()
            elif offset > 0 and line_start == previous_line and not current:
                continue
            contiguous = (
                bool(current)
                and paragraph == current_paragraph
                and sentence.location.char_start - previous_end <= 2
            )
            if not contiguous and current:
                flush()
            current = f"{current} {piece.strip()}" if contiguous else piece.strip()
            current_paragraph = paragraph
            previous_end = sentence.location.char_end
            previous_line = line_start

    flush()
    return merged


#: A reference is finished once it has a year and the punctuation density of a
#: bibliographic record: "Abell, G. O. 1958, ApJS, 3, 211" has four commas. A
#: wrapped continuation line rarely does.
_COMPLETE_ENTRY_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _looks_complete(text: str) -> bool:
    """Whether ``text`` already reads as a whole bibliographic entry."""
    stripped = text.strip()
    if len(stripped) < 25:
        return False
    return bool(_COMPLETE_ENTRY_RE.search(stripped)) and stripped.count(",") >= 3


#: A new entry begins with a ``Surname, X.`` / ``Surname X.`` author block. A
#: title essentially never has that shape, so requiring it keeps
#: ``...yöntemleri. Eğitim Bilimleri Dergisi...`` in one piece while still
#: separating two references someone pasted into one paragraph.
_NEW_ENTRY_RE = re.compile(r"(?<=\.)\s+(?=[A-ZÇĞİÖŞÜ][a-zçğıöşüà-ÿ]{2,},\s*[A-ZÇĞİÖŞÜ]\.)")


def _split_entry(text: str) -> list[str]:
    """Split a merged paragraph into individual bibliography entries."""
    return [p for p in _NEW_ENTRY_RE.split(text) if len(p.strip()) > 20]


def _vet(text: str, index: int, this_year: int) -> ReferenceEntry:
    flags: list[str] = []
    doi = _DOI_RE.search(text)
    arxiv = _ARXIV_RE.search(text)
    url = _URL_RE.search(text)
    identifier = identifier_kind = None
    if doi:
        identifier, identifier_kind = doi.group(0).rstrip(".,;"), "doi"
    elif arxiv:
        identifier = f"arXiv:{arxiv.group(1)}{arxiv.group(2)}.{arxiv.group(3)}"
        identifier_kind = "arxiv"
    elif url:
        identifier, identifier_kind = url.group(0).rstrip(".,;"), "url"

    # Year extraction is positional, not "the largest number present".
    #
    # Measured on the real bibliography of a 154-page astrophysics thesis
    # (arXiv:1407.6566, 189 entries): taking ``max(years)`` flagged nine entries
    # as problems when none was. That bibliography is in astronomy style,
    # ``Author. Year, Journal, Volume, Page [back-references]``, so
    # "MNRAS, 403, 2063 11" reads 2063 as a year - it is a page number - and
    # "J. 1958, ApJS, 3, 211 1, 4, 10" would otherwise be outranked by nothing,
    # while a back-reference like "571 4" could outrank the real year.
    #
    # The publication year sits between the authors and the venue, so it is the
    # first four-digit number that is followed by a comma. Only if that fails does
    # the scan fall back to any year-shaped number.
    author_position = _AUTHOR_YEAR_RE.search(text)
    year: int | None
    if author_position is not None:
        year = int(author_position.group(1))
    else:
        years = [int(y) for y in _YEAR_RE.findall(text)]
        year = years[0] if years else None

    has_authors = bool(_INITIAL_RE.search(text)) or len(text.split()) >= 4
    has_venue = bool(_VENUE_HINT_RE.search(text))

    # A DOI written but not formed is a signature of a generated bibliography.
    if identifier is None and _DOI_LOOSE_RE.search(text):
        flags.append("doi_yazilmis_ama_gecersiz")
    if _ARXIV_RE.search(text) is None and re.search(r"arxiv[:\s/]*\d{2}\d{2}", text, re.I):
        flags.append("arxiv_kimligi_bozuk")
    if identifier is None:
        flags.append("kalici_kimlik_yok")
    if year is None:
        flags.append("yil_yok")
    elif year > this_year:
        flags.append("gelecek_yil")
    elif year < MIN_PLAUSIBLE_YEAR:
        flags.append("imkansiz_eski_yil")
    if not has_authors:
        flags.append("yazar_bilgisi_zayif")
    if not has_venue:
        flags.append("yayin_yeri_belirsiz")
    if _SENTENCE_LIKE_RE.search(text):
        flags.append("cumle_yapisinda")
    if len(text) > 400:
        flags.append("asiri_uzun")

    risk = _rate(flags)
    return ReferenceEntry(
        index=index,
        text=text,
        year=year,
        has_identifier=identifier is not None,
        identifier=identifier,
        identifier_kind=identifier_kind,
        has_authors=has_authors,
        has_venue=has_venue,
        flags=tuple(flags),
        risk=risk,
    )


def _rate(flags: Sequence[str]) -> str:
    """Escalate only on a cluster, never on one missing field.

    The measured shape of fabrication is a *cluster* (no identifier, no year, no
    venue, broken DOI), and the measured shape of a legitimate print-only
    reference is a single missing identifier. One signal is a note; three is a
    review.
    """
    severe = {
        "doi_yazilmis_ama_gecersiz",
        "arxiv_kimligi_bozuk",
        "gelecek_yil",
        "imkansiz_eski_yil",
    }
    strong = {"kalici_kimlik_yok", "yil_yok", "yayin_yeri_belirsiz"}
    severe_hits = len(severe & set(flags))
    strong_hits = len(strong & set(flags))
    if severe_hits or strong_hits >= 3:
        return "review"
    if strong_hits >= 2 or "yazar_bilgisi_zayif" in flags:
        return "note"
    return "ok"