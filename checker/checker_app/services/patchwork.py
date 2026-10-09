"""Patchwriting: reuse assembled from many short spans instead of one long one.

Why this module exists
----------------------
A similarity percentage answers "how much text overlaps". It does not answer
"how the overlap is *built*", and the two questions come apart exactly where a
thesis is at risk. Copying two paragraphs is one incident. Copying forty
eight-word sentences from thirty papers is a different one, and it is the classic
thesis failure mode: the document is locally legitimate and globally assembled.

This is the quantity-sensitive family of measures. The design follows
Meuschke, Stange, Schubotz, Kramer & Gipp, "HyPlag: Hybrid Plagiarism Detection
by Incorporating Citation Information" (JCDL 2019, arXiv:1906.11761), whose
*Greedy Identifier Tiles* (GIT) counts **how many reused units** a document
carries rather than how long the single longest run is. In HyPlag the units are
mathematical identifiers or citations; here they are words, so the same
construction becomes a word-level patchwork score.

Measured evidence this design follows (all from arXiv:1906.11761):

* **Quantity sensitivity pays.** Against 10 confirmed retracted-for-plagiarism
  STEM papers, the order-aware, quantity-sensitive ``GIT`` reached MRR **0.79**
  on math identifiers while the order-agnostic ``Histo`` reached **0.58**. Counting
  ordered reused *units* beat counting shared vocabulary. Caveat: n=10, so the
  confidence interval is wide.
* **Chance baselines are measurable, so we publish ours.** HyPlag measured
  significance thresholds over **1,000,000 random document pairs**: ``GIT`` ≥
  **0.15**, ``GCT`` ≥ **0.10**, ``Encoplot`` (16-gram containment) ≥ **0.06**,
  ``BC`` ≥ 0.13, ``LCCS`` ≥ 0.22. Those are the only corpus-derived thresholds in
  this whole literature; every institutional percentage is a locally chosen
  number with no derivation behind it.
* **Cross-corpus transfer fails.** The same detector scored plagdet 0.61 on
  PAN-2025 and **0.17** on PAN-2012 (arXiv:2510.06805). A score without its
  chance baseline cannot be read.

Two corrections to common assumptions, both measured:

* **"PPBS / patchwork plagiarism score" does not exist.** No traceable definition,
  range or AUC appears in the peer-reviewed literature; searches resolve to
  unrelated physics/biology papers and to vendor marketing. HyPlag's GIT is the
  real, citable quantity-sensitive measure and is what this module implements.
* **Many short matches are not automatically worse than few long ones.** COPE's
  concentration principle states that duplication *spread across many short
  phrases may be less concerning* than duplication concentrated in a few
  paragraphs/sections (COPE, *Determining Acceptable Levels of
  Plagiarism/Duplication*). So this module reports the score **and** the shape:
  a document with twenty five-word tiles reads differently from one with two
  three-hundred-word blocks, and the report says which it has.

What this module does not do
----------------------------
It does not call anything misconduct. Paraphrased reuse is under-detected by
construction: on PAN-2025, naive embedding baselines flagged *genuine* LLM-
paraphrased text roughly **twice** as often as actual plagiarism, and the best
system reached precision 0.58 at recall 0.82 (arXiv:2510.06805). Humans detect
machine-paraphrased plagiarism at only **53%** accuracy (Wahle et al., EMNLP
2022, 10.18653/v1/2022.emnlp-main.62). This score is a shape descriptor for a
human reviewer.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from checker_app.domain.enums import CitationStatus
from checker_app.domain.models import MatchSpan
from checker_app.domain.text import Token
from checker_app.services.attribution import Span

__all__ = [
    "PatchworkReport",
    "PatchworkTile",
    "CHANCE_BASELINES",
    "MIN_TILE_WORDS",
    "build_patchwork_report",
]

#: HyPlag's ``GIT`` requires at least 5 matching identifiers per tile. The same
#: floor is applied here: a "tile" of one or two shared words is ordinary
#: academic phrasing, and a measure that counts it measures coincidence.
MIN_TILE_WORDS = 5


@dataclass(frozen=True, slots=True)
class PatchworkTile:
    """One greedily-accepted run of reused words, and where it came from."""

    source_id: str
    source_name: str
    words: int
    doc_char_start: int
    doc_char_end: int
    source_char_start: int
    source_char_end: int
    quoted: bool = False
    attributed: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source_name,
            "words": self.words,
            "character_range": [self.doc_char_start, self.doc_char_end],
            "source_range": [self.source_char_start, self.source_char_end],
            "quoted": self.quoted,
            "attributed": self.attributed,
        }


@dataclass(frozen=True, slots=True)
class PatchworkReport:
    """The score, the chance baseline it must be read against, and its shape."""

    score: float
    """Greedy Identifier Tiles normalised by document words - the direct analogue
    of ``s_GIT = ||T_l|| / (I_d - 1)``. 0 means no tile was long enough to count."""

    tile_count: int
    """``||T_l||``: how many distinct reused runs survived the greedy pass. This is
    the quantity-sensitive part; a 60-word paragraph and twelve 5-word fragments
    both score 1.0 but are not the same incident."""

    tile_words: int
    matched_words: int
    """All matched words, including fragments too short to tile."""

    document_words: int
    largest_tile_words: int
    mean_tile_words: float
    quoted_tile_words: int
    unattributed_tile_words: int
    tiles: tuple[PatchworkTile, ...] = ()
    per_source: tuple[tuple[str, int], ...] = ()
    chance_baseline: float = 0.15
    """HyPlag's measured significance threshold for ``GIT`` over 1,000,000 random
    document pairs. A score below this is not distinguishable from chance."""

    method: str = (
        "Greedy Identifier Tiles (HyPlag, JCDL 2019, arXiv:1906.11761): accept the "
        "longest reused run first, keep it only if it does not overlap an already "
        "accepted one, drop runs shorter than "
        f"{MIN_TILE_WORDS} words, divide the count by document words."
    )
    reading: str = ""
    shape: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def above_chance(self) -> bool:
        return self.score >= self.chance_baseline

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "tile_count": self.tile_count,
            "tile_words": self.tile_words,
            "matched_words": self.matched_words,
            "document_words": self.document_words,
            "largest_tile_words": self.largest_tile_words,
            "mean_tile_words": round(self.mean_tile_words, 2),
            "quoted_tile_words": self.quoted_tile_words,
            "unattributed_tile_words": self.unattributed_tile_words,
            "chance_baseline": self.chance_baseline,
            "above_chance": self.above_chance,
            "method": self.method,
            "reading": self.reading,
            "shape": self.shape,
            "caveats": list(self.caveats),
            "per_source": [
                {"source": name, "tiles": tiles} for name, tiles in self.per_source
            ],
            "tiles": [t.to_dict() for t in self.tiles],
        }


#: Significance thresholds measured by HyPlag over 1,000,000 random document
#: pairs (common-author and citation pairs excluded). Published so a reader can
#: see that the 0.15 used here is a measurement, not a taste.
CHANCE_BASELINES: dict[str, float] = {
    "GIT (greedy identifier tiles, math)": 0.15,
    "GCT (greedy citation tiles)": 0.10,
    "Encoplot (shared character 16-grams)": 0.06,
    "BC (bibliographic coupling)": 0.13,
    "LCCS (longest common citation sequence)": 0.22,
    "Histo": 0.56,
    "LCIS (longest common identifier sequence)": 0.76,
}


def build_patchwork_report(
    *,
    tokens: Sequence[Token],
    matches: Sequence[MatchSpan],
    total_words: int,
    quote_spans: Sequence[Span] = (),
    statuses: dict[int, CitationStatus] | None = None,
    min_tile_words: int = MIN_TILE_WORDS,
) -> PatchworkReport:
    """Tile the matches greedily and describe the resulting shape.

    ``statuses`` maps ``id(match)`` -> :class:`CitationStatus`. Attribution does
    not change the score - reuse is reuse - but it changes what the score means,
    so quoted and unattributed words are counted separately.
    """
    statuses = statuses or {}
    starts = [token.char_start for token in tokens]
    quote_indices = _indices_in_spans(starts, quote_spans)
    denominator = max(1, total_words - 1)  # HyPlag's I_d - 1

    matched_words = sum(m.matched_words for m in matches)

    # Greedy: longest first, each tile keeps only the words no earlier tile took.
    # HyPlag does the same - a tile is a *tile*, not a second copy of its
    # neighbour, which is what stops one paragraph being counted five times.
    ordered = sorted(matches, key=lambda m: (-m.matched_words, m.doc_char_start))
    used: set[int] = set()
    tiles: list[PatchworkTile] = []
    for match in ordered:
        if match.matched_words < min_tile_words:
            continue
        indices = _indices_in_range(starts, match.doc_char_start, match.doc_char_end)
        if not indices or indices & used:
            continue
        status = statuses.get(id(match))
        tiles.append(
            PatchworkTile(
                source_id=match.source_id,
                source_name=match.source_name,
                words=match.matched_words,
                doc_char_start=match.doc_char_start,
                doc_char_end=match.doc_char_end,
                source_char_start=match.source_char_start,
                source_char_end=match.source_char_end,
                quoted=bool(indices & quote_indices),
                attributed=status is CitationStatus.CITED_AND_QUOTED,
            )
        )
        used |= indices

    tile_words = sum(t.words for t in tiles)
    quoted = sum(t.words for t in tiles if t.quoted)
    unattributed = sum(t.words for t in tiles if not t.attributed)
    largest = max((t.words for t in tiles), default=0)
    mean_tile = tile_words / len(tiles) if tiles else 0.0
    score = len(tiles) / denominator

    per_source: dict[str, int] = {}
    for tile in tiles:
        per_source[tile.source_name] = per_source.get(tile.source_name, 0) + 1
    ordered_sources = tuple(sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0])))

    report = PatchworkReport(
        score=score,
        tile_count=len(tiles),
        tile_words=tile_words,
        matched_words=matched_words,
        document_words=total_words,
        largest_tile_words=largest,
        mean_tile_words=mean_tile,
        quoted_tile_words=quoted,
        unattributed_tile_words=unattributed,
        tiles=tuple(tiles),
        per_source=ordered_sources,
    )
    return _with_reading(report)


def _with_reading(report: PatchworkReport) -> PatchworkReport:
    """Attach the two sentences that decide what the number means."""
    caveats = [
        "Bu puan suistimal değildir; insan gözden geçirme için bir biçim betimleyicisidir.",
        "Yeniden ifade edilmiş (paraphrase) kullanım bu ölçütün dışında kalır: PAN-2025'te "
        "en iyi sistem precision 0.58 / recall 0.82, naif gömme tabanlı yöntemler gerçek "
        "metni uydurulmuş intihalin ~2 katı oranında işaretledi (arXiv:2510.06805); "
        "makine yeniden ifadesi intihali insanlar %53 doğrulukla buluyor "
        "(10.18653/v1/2022.emnlp-main.62).",
        "Eşikler korpusa göre kurulur: aynı yöntem PAN-2025'te plagdet 0.61, PAN-2012'de "
        "0.17 verdi (arXiv:2510.06805). Bu yüzden puan şans tabanıyla birlikte "
        "okunmalıdır.",
    ]
    if report.tile_count == 0:
        return replace(
            report,
            reading=(
                f"{MIN_TILE_WORDS} kelimelik bir karo oluşmadı: eşleşmeler ya daha kısa ya "
                "da alıntı içinde. Bu, 'temiz' demek değildir — yeniden ifade edilmiş "
                "kullanım buradaki tüm ölçütlerin dışında kalır."
            ),
            shape="yok",
            caveats=tuple(caveats),
        )

    ratio = report.score / report.chance_baseline
    if report.tile_count < 3:
        # COPE's concentration principle compares *distributions* of reuse. With
        # one or two tiles there is no distribution, and calling a single reused
        # paragraph "spread" would be a category error.
        shape = "belirsiz"
        shape_note = (
            f"{report.tile_count} karo var; COPE'nin yoğunlaşma ilkesi bir dağılım "
            "karşılaştırdığı için tek ya da iki karo için biçim okuması yapılmaz."
        )
    elif report.mean_tile_words < 20:
        shape = "yayılmış"
        shape_note = (
            f"ortalama karo {report.mean_tile_words:.1f} kelime: metin çok sayıda kısa "
            "parçadan derlenmiş. COPE'ye göre birçok kısa ifadeye yayılmış tekrar, "
            "birkaç paragrafa yoğunlaşmış tekrardan *daha az* endişe vericidir "
            "(COPE, Determining Acceptable Levels of Plagiarism/Duplication)."
        )
    else:
        shape = "yoğunlaşmış"
        shape_note = (
            f"ortalama karo {report.mean_tile_words:.1f} kelime, en büyüğü "
            f"{report.largest_tile_words} kelime: tekrar birkaç büyük blokta toplanmış. "
            "Bu, COPE'nin yoğunlaşma ilkesinde daha endişe verici olan biçimdir."
        )

    return replace(
        report,
        reading=(
            f"{report.tile_count} karo / {report.document_words} kelime = "
            f"{report.score:.4f}; şans tabanı {report.chance_baseline} → "
            + (
                f"tabanın {ratio:.1f} katı."
                if report.above_chance
                else f"tabanın altında ({ratio:.2f}×), şansa ayırt edilemiyor."
            )
            + (
                f" Karoların {report.unattributed_tile_words} kelimesi atıfsız, "
                f"{report.quoted_tile_words} kelimesi alıntı içinde."
                if report.unattributed_tile_words or report.quoted_tile_words
                else " Tüm karolar atıflı."
            )
        ),
        shape=f"{shape} — {shape_note}",
        caveats=tuple(caveats),
    )


# --------------------------------------------------------------------------- utils
def _indices_in_range(
    starts: Sequence[int], char_start: int, char_end: int
) -> set[int]:
    """Token indices whose text falls inside ``[char_start, char_end)``."""
    # Tokens are emitted in document order, so this is a slice, not a scan. The
    # bisect matters: a 20 000-word thesis with 900 matches is 18M comparisons
    # without it, which shows up in every report.
    return set(range(bisect_left(starts, char_start), bisect_left(starts, char_end)))


def _indices_in_spans(starts: Sequence[int], spans: Sequence[Span]) -> set[int]:
    out: set[int] = set()
    for span in spans:
        out |= _indices_in_range(starts, span.char_start, span.char_end)
    return out