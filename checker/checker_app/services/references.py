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
#: A reference that predates the discipline's plausible start is as suspicious as
#: one dated in the future, and both are cheap to spot offline.
MIN_PLAUSIBLE_YEAR = 1950
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
        return {
            "references_scanned": self.total,
            "needs_review": len(self.flagged),
            "with_identifier": sum(1 for e in self.entries if e.has_identifier),
            "without_identifier": sum(1 for e in self.entries if not e.has_identifier),
            "without_year": sum(1 for e in self.entries if e.year is None),
            "risk_counts": {
                level: sum(1 for e in self.entries if e.risk == level)
                for level in ("ok", "note", "review")
            },
            "flag_counts": counts,
            "scanned_section": self.scanned_section,
        }

    def to_dict(self, limit: int = 40, *, only_flagged: bool = True) -> dict[str, object]:
        payload = self.summary()
        rows = self.flagged if only_flagged else self.entries
        payload["entries"] = [e.to_dict() for e in rows[: max(1, limit)]]
        payload["truncated"] = len(rows) > max(1, limit)
        payload["evidence"] = list(self.evidence)
        payload["caveats"] = [
            "Bu denetim yalnızca YAPISALDIR: DOI çözülmez, CrossRef sorgulanmaz, PDF okunmaz. "
            "Bir kaydın gerçekten var olduğunu kanıtlayamaz.",
            "Tek başına kaynak denetimi suistimal kanıtı değildir. Doğrulanamayan bir kayıt, "
            "çoğu bulunmayan meşru bir kaynak da olabilir; tez oranları için yayımlanmış "
            "ölçüm yoktur (yukarıdaki tüm sayılar makale/konferans kayıtlarıdır).",
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

    for sentence in lines:
        if sentence.location.block_type.value == "heading":
            if current:
                merged.append((current, previous_end))
                current = ""
            current_paragraph = None
            previous_end = -1
            continue
        paragraph = sentence.location.paragraph_index
        contiguous = (
            bool(current)
            and paragraph == current_paragraph
            and sentence.location.char_start - previous_end <= 2
        )
        if not contiguous and current:
            merged.append((current, previous_end))
        current = f"{current} {sentence.text}" if contiguous else sentence.text
        current_paragraph = paragraph
        previous_end = sentence.location.char_end

    if current:
        merged.append((current, previous_end))
    return merged


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

    years = [int(y) for y in _YEAR_RE.findall(text)]
    year = max(years) if years else None

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