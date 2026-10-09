"""MCP (Model Context Protocol) server exposing the thesis checker.

Runs **in-process** against the same :class:`~checker_app.services.runner.ScanRunner`
the CLI uses, so an assistant can ask "where in this thesis is the risk, and why"
without a second process.

    checker mcp                       # stdio (Claude Desktop, Claude Code, …)
    checker mcp --http --port 8090     # streamable HTTP for remote clients

Design notes
------------
* **Compact payloads.** A thesis report is thousands of sentences; dumping it
  into a context window is the fastest way to waste a model's attention. Every
  tool returns a summary plus the top findings and points at a follow-up tool
  for the rest (``checker_sentences``, ``checker_matches``).
* **Located findings, always.** Sentence index, character offsets, line/page/
  section and a plain-language reason travel together, because "this sentence is
  risky" without a location is useless to a reviewer.
* **Read-only hints.** The scan tools only read files the caller names and never
  write, so they are annotated ``read_only_hint`` and clients may auto-approve.
  ``checker_calibrate`` reads a corpus but needs model weights, so it is not.
* **Degradation is reported, never hidden.** When a signal cannot run (a missing
  model, an English-only classifier on Turkish text) the payload says so in
  ``degraded_signals``; the tool never silently returns a clean report.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from checker_app import __version__
from checker_app.config import CheckerSettings, get_settings
from checker_app.domain.enums import RiskLevel
from checker_app.logging import configure_logging, get_logger, new_id
from checker_app.services.calibration import calibrate as calibrate_corpus
from checker_app.services.compliance import build_compliance_report
from checker_app.services.discourse import RELATION_DIRECTION
from checker_app.services.patchwork import CHANCE_BASELINES
from checker_app.services.runner import ScanRequest, ScanRunner
from checker_app.services.scoring import (
    AI_DEGREE_CAVEAT,
    AI_HUMAN_BASELINE,
    AI_TURKISH_FPR,
    AI_TURKISH_FPR_SOURCE,
)
from checker_app.services.sources import SourceLoader

logger = get_logger(__name__)

INSTRUCTIONS = """\
Academic thesis checker: sentence-level AI-authorship risk and plagiarism.

Typical flow:
  1. `checker_segment` to see how a document is cut into located sentences -
     fast, no model weights needed. Use this to check the segmentation before
     spending time on a full scan.
  2. `checker_scan` for the full analysis: per-sentence AI risk plus plagiarism
     against the reference sources you pass. Start with `min_level="medium"` and
     raise it only if the thesis looks risky; `top` bounds the payload.
  3. `checker_similarity` for the institutional numbers (the percentages a thesis
     committee actually reads) and `checker_compliance` for the YOK/institutional
     checklist: disclosure, attribution status, AI-heavy sections.
  4. `checker_sentences` and `checker_matches` to drill into what the summary
     flagged. Both return character offsets and section roles.
  5. `checker_patchwork` when the question is "how is this text assembled?"
     rather than "how much overlaps" - many short reused runs read differently
     from a few long ones, and COPE treats the concentrated form as more
     concerning.
  6. `checker_references` for the bibliography. It is a structural audit only,
     so treat it as a list of entries a human should look up by hand, never as
     proof that a source does not exist.

Rules worth knowing before you trust a number:
  - These are risk *estimates*, never proof of authorship. The benchmark authors
    themselves oppose detector use in disciplinary contexts, and measured false
    positive rates for some detectors on formal academic writing reach 61-80%.
  - **Never say how much of a text was machine written.** Editing only 1% of a
    human text raises GLTR's false-positive rate from 6.83% to 26.85%, and
    detectors cannot grade the degree at all (APT-Eval, arXiv:2502.15666).
  - **Below a 20% AI share there is no document-level verdict**, only located
    sentence findings. Turnitin's AIW-2 protocol does the same, and measures
    document FPR 0.51% vs sentence FPR 0.33% on 719,877 pre-2019 human writings.
  - Lecturers are not better at this than machines: 63 of them read German thesis
    excerpts at 57% (AI) and 64% (human) accuracy, under 20% on professional-level
    AI text (10.1016/j.iree.2025.100321).
  - `cited_and_quoted` matches are legitimate reuse. Only 6.4% of matched pairs
    in a 2025-26 corpus study carried a citation at all, so a match count is not
    a misconduct count.
  - YOK sets no similarity percentage; each institute board does. The common
    Turkish filters are "bibliography excluded, quotations included, matches under
    5 words excluded".
  - **The two languages have different baselines, by roughly 3x.** Turkish theses
    measure 28.7% +/- 11.58 similarity (n=600); clean English dissertations
    measure **9% +/- 6%** (n=360). `checker_similarity` picks the baseline and the
    institutional bands from the detected document language. For English the
    *measured* optimal cutoff is **15%** (sensitivity 84.8%, specificity 80.5%,
    AUC 0.902); 25-30% is a policy ceiling, not an observed normal.
  - **English AI prevalence is field-dependent.** arXiv CS abstracts 22.5%
    versus mathematics 7.7% (introductions 4.1%), estimator error under 3.5
    points. Mathematics is the one field where introductions are *less* modified
    than abstracts, so reading it against the CS figure inverts the prior.
  - **English thresholds must be section- and length-banded.** Detector accuracy
    on academic prose runs 0.86 humanities to 0.51 science, and 0.87 at 300-330
    words to 0.56 at 450-550 (Hadra et al. 2026). A single document-level English
    score is wrong.
  - Turkish detectors fail on Turkish academic prose, measured. Altıntop (2026,
    DOI 10.56493/nkusbmyo.1866431) ran eight detectors on a 5,715-word Turkish
    academic text written with no AI at all: Justdone called it 89% AI, ZeroGPT
    ~80%, TruthScan 40%. The same paper's cross-language check is why this tool
    ships its own operating point: on the identical passage of Derrida's *Plato's
    Pharmacy*, ZeroGPT read **0% AI** on the French original and **73.25% AI** on
    the Turkish translation. Published Turkish AUROC/FPR figures do exist (AUROC
    99.31%, FPR 5.84%, DOI 10.28948/ngumuh.1930411) but on news/abstract/homework
    registers, where the two closest to a thesis score lowest - academic
    95.21%, official/legal 94.99%.
  - The bundled classifier is English-only, so it is dropped automatically on
    Turkish with a reason in `degraded_signals`.
  - Reuse detection collapses under translation: measured precision is **80%** on
    untranslated text, **26.7%** on translated text and **16.7%** on
    translated-then-paraphrased text (DOI 10.33806/ijaes1026). A Turkish thesis
    citing English or German literature sits in the worst of those rows, so a
    low similarity percentage is partly a statement about the corpus you
    supplied, not about the thesis.
  - **Turkish detectors fail on Turkish academic prose, measured.** Altıntop
    (2026, DOI 10.56493/nkusbmyo.1866431) ran eight detectors on a 5,715-word
    Turkish academic text written with no AI at all: Justdone called it 89% AI,
    ZeroGPT ~80%, TruthScan 40%. The same paper's cross-language check is the
    reason this tool ships its own operating point: on the identical passage of
    Derrida's *Plato's Pharmacy*, ZeroGPT read **0% AI** on the French original
    and **73.25% AI** on the Turkish translation. Published Turkish AUROC/FPR
    figures do exist (AUROC 99.31%, FPR 5.84%, DOI 10.28948/ngumuh.1930411) but
    on news/abstract/homework registers, where the two closest to a thesis score
    lowest - academic 95.21%, official/legal 94.99%.
"""

#: Tool annotations, mirroring the paper-app conventions.
_READ_ONLY_LOCAL = {"read_only_hint": True, "open_world_hint": False, "destructive_hint": False}
_NEEDS_MODELS = {"read_only_hint": True, "open_world_hint": True, "destructive_hint": False}

_MAX_ITEMS = 50


# --------------------------------------------------------------------------- setup
@asynccontextmanager
async def _lifespan(server: Any) -> AsyncIterator[CheckerSettings]:  # noqa: ANN401
    """Own the settings/logging for the server's lifetime."""
    settings = get_settings()
    configure_logging(settings.log_level)
    new_id()
    try:
        yield settings
    finally:
        logger.info("checker mcp kapanıyor")


def _build_server() -> Any:  # noqa: ANN401
    from mcp.server.mcpserver import MCPServer  # noqa: PLC0415

    return MCPServer(
        name="thesis-checker",
        title="Thesis Checker",
        version=__version__,
        instructions=INSTRUCTIONS,
        lifespan=_lifespan,
    )


server = _build_server()


def _ann(**kwargs: Any):  # noqa: ANN202
    from mcp.types import ToolAnnotations  # noqa: PLC0415

    return ToolAnnotations(**kwargs)


# --------------------------------------------------------------------------- utils
def _settings() -> CheckerSettings:
    return get_settings()


def _level(value: str | None) -> RiskLevel | None:
    if not value:
        return None
    try:
        return RiskLevel(value)
    except ValueError:
        return None


def _paths(paths: list[str] | None) -> list[Path]:
    return [Path(p).expanduser() for p in (paths or [])]


def _load_sources(
    loader: SourceLoader,
    refs: list[str] | None,
    ref_dirs: list[str] | None,
    ref_glob: str,
    exclude: list[str] | None,
    exclude_document: Path | None,
) -> tuple[list[Any], list[str]]:
    """Load reference documents, skipping the thesis and its own reports."""
    loaded = [loader.load_reference(spec) for spec in (refs or [])]
    for directory in ref_dirs or []:
        loaded.extend(loader.load_directory(directory, ref_glob))
    dropped: list[str] = []
    kept = []
    for item in loaded:
        name = Path(item.path)
        if exclude_document is not None and item.kind in ("file", "dir"):
            # Note: directory loads are kind="dir", so an earlier `kind == "file"`
            # guard silently disabled this for the common --ref-dir case and the
            # thesis was compared with itself at 100%.
            try:
                if name.resolve() == exclude_document.resolve():
                    dropped.append(f"{name} (belgenin kendisi)")
                    continue
            except OSError:  # pragma: no cover - defensive
                pass
        if any(name.match(g) for g in (exclude or [])):
            dropped.append(f"{name} (--exclude)")
            continue
        kept.append(item)
    return kept, dropped


def _sentence_payload(sentence, *, with_findings: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "index": sentence.index,
        "text": sentence.snippet(240),
        "location": sentence.location.label(),
        "character_range": [sentence.location.char_start, sentence.location.char_end],
        "line_range": [sentence.location.line_start, sentence.location.line_end],
        "page": sentence.location.page,
        "section": " › ".join(sentence.location.section_path) or "(giriş)",
        "section_role": sentence.section_role.value,
        "words": sentence.word_count,
        "risk": sentence.risk.value,
        "ai_score": sentence.ai_score,
        "plagiarism_score": sentence.plagiarism_score,
        "confidence": sentence.confidence,
        "ai_percent": sentence.ai_percent,
    }
    if sentence.ratio:
        out["likelihood_ratio"] = {
            "score": sentence.ratio.score,
            "observer": sentence.ratio.observer_model,
            "performer": sentence.ratio.performer_model,
        }
    if sentence.perplexity:
        out["perplexity"] = sentence.perplexity.perplexity
    if sentence.classifier:
        out["classifier_p_ai"] = sentence.classifier.p_ai
    if sentence.plagiarism and sentence.plagiarism.matches:
        out["matches"] = [
            {
                "kind": m.kind.value,
                "source": m.source_name,
                "matched_words": m.matched_words,
                "character_range": [m.doc_char_start, m.doc_char_end],
                "source_range": [m.source_char_start, m.source_char_end],
                "citation_status": (
                    m.citation_status.value if m.citation_status else None
                ),
                "snippet": m.snippet[:160],
            }
            for m in sentence.plagiarism.matches[:5]
        ]
    if with_findings and sentence.findings:
        out["reasons"] = [
            {"message": f.message, "evidence": f.evidence} for f in sentence.findings[:4]
        ]
    return out


def _style_context(report) -> dict[str, Any]:  # noqa: ANN401
    """Document-level context that is reported but never scored.

    Each block exists because a number without its caveat gets over-read:
    lexical richness has an unstable sign (light AI editing raises it above the
    human value), and word count alone reaches ROC-AUC 0.68.
    """
    out: dict[str, Any] = {}
    if report.discourse is not None:
        out["discourse"] = report.discourse.to_dict()
    stats = report.style_stats
    if stats is not None:
        out["lexical_richness"] = (
            stats.lexical_richness.to_dict() if stats.lexical_richness else None
        )
        out["length_confound"] = (
            stats.length_confound.to_dict() if stats.length_confound else None
        )
    return out


def _verdict(report) -> dict[str, Any]:  # noqa: ANN001
    return {
        "risk": report.risk.value,
        "ai_score": report.ai_score,
        "plagiarism_score": report.plagiarism_score,
        "flagged_sentences": report.flagged_sentences,
        "high_risk_sentences": report.high_risk_sentences,
        "matched_sentences": report.matched_sentences,
        "unscored_sentences": report.unscored_sentences,
        "mean_perplexity": report.mean_perplexity,
        "perplexity_burstiness": report.perplexity_burstiness,
        "ai_share": report.ai_share,
        "ai_flagged_words": report.ai_flagged_words,
        "below_ai_reporting_floor": report.below_ai_reporting_floor,
        "style_context": _style_context(report),
        "english_context": report.english_context.to_dict() if report.english_context else None,
        "document_ai_claim": None if report.below_ai_reporting_floor else report.ai_score,
        "document_ai_claim_note": (
            "AI kapsamı %20'nin altında: belge düzeyinde AI iddiası üretilmiyor, "
            "yalnız cümle düzeyi bulgular var. Turnitin AIW-2 protokolü de "
            "719.877 insan metninde belge FPR %0.51 / cümle FPR %0.33 ölçtüğü için "
            "böyle bir ayrım yapıyor."
            if report.below_ai_reporting_floor
            else "AI kapsamı raporlama tabanının üstünde."
        ),
        "models": report.models,
        "degraded_signals": list(report.degraded_signals),
        "caveats": {
            "degree": report.degree_caveat,
            "human_baseline": report.human_baseline,
            "turkish_fpr": AI_TURKISH_FPR,
            "turkish_fpr_source": AI_TURKISH_FPR_SOURCE,
        },
    }


def _document_summary(report) -> dict[str, Any]:  # noqa: ANN001
    return {
        "path": report.path,
        "language": report.language,
        "language_confidence": round(report.language_confidence, 3),
        "sentences": report.sentence_count,
        "words": report.word_count,
        "pages": report.page_count,
        "duration_seconds": report.duration_seconds,
    }


def _sections(report, limit: int = 12) -> list[dict[str, Any]]:  # noqa: ANN001
    return [
        {
            "section": s.title,
            "sentences": len(s.sentence_indices),
            "ai_score": s.ai_score,
            "plagiarism_score": s.plagiarism_score,
            "high_risk_sentences": s.high_risk_count,
        }
        for s in report.sections[:limit]
    ]


def _run_scan(
    settings: CheckerSettings,
    path: str,
    *,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    ref_glob: str = "*",
    exclude: list[str] | None = None,
    language: str | None = None,
    profile: str = "document",
    min_level: str | None = None,
    top: int = 25,
    use_ratio: bool = True,
    use_perplexity: bool = True,
    use_classifier: bool = True,
    use_code: bool = True,
    ppl_model: str | None = None,
    classifier_model: str | None = None,
    ratio_model: str | None = None,
):  # noqa: ANN201, PLR0913
    """Shared scan pipeline for the scan-like tools."""
    if profile in ("document", "thesis"):
        settings.profile = profile  # type: ignore[assignment]
    loader = SourceLoader(settings)
    document = Path(path).expanduser()
    loaded = loader.load_document(document)
    references, dropped = _load_sources(
        loader, refs, ref_dirs, ref_glob, exclude, document
    )
    runner = ScanRunner(settings, sources=loader.to_sources(references))
    report = runner.scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            language=None if language in (None, "auto") else language,
            use_ratio=use_ratio,
            use_perplexity=use_perplexity,
            use_classifier=use_classifier,
            use_plagiarism=bool(references),
            use_code=use_code,
            perplexity_model=ppl_model,
            classifier_model=classifier_model,
            ratio_model=ratio_model,
        )
    )
    level = _level(min_level)
    flagged = [
        s
        for s in report.sentences
        if level is None or s.risk.at_least(level)
    ]
    flagged.sort(key=lambda s: (-s.risk_score, s.index))
    return report, loaded, references, dropped, flagged[: max(1, top)]


# --------------------------------------------------------------------------- tools
@server.tool(
    name="checker_segment",
    title="Segment a document",
    description=(
        "Split a document into located sentences without loading any model. Fast and "
        "offline: use it to verify how the text is cut (which sentence is #42, where it "
        "sits) before running a full scan, or to tune segmentation options."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_segment(
    path: str,
    min_level: str | None = None,
    max_items: int = 60,
    include_all: bool = False,
) -> dict[str, Any]:
    """Sentence segmentation with character/line/page/section locations."""
    from checker_app.domain.segmentation import SentenceSplitter  # noqa: PLC0415

    settings = _settings()
    loaded = SourceLoader(settings).load_document(path)
    result = SentenceSplitter(settings.segmentation).split(loaded.text)

    items = [
        {
            "index": s.index,
            "text": s.snippet(200),
            "character_range": [s.location.char_start, s.location.char_end],
            "line_range": [s.location.line_start, s.location.line_end],
            "page": s.location.page,
            "paragraph": s.location.paragraph_index,
            "section": " › ".join(s.location.section_path) or "(giriş)",
            "section_role": s.location.section_role.value,
            "block_type": s.location.block_type.value,
            "words": s.word_count,
        }
        for s in result.sentences
    ]
    if not include_all:
        level = _level(min_level)
        if level is not None:
            items = [
                item
                for item, s in zip(items, result.sentences, strict=True)
                if s.location.char_end  # keep the mapping honest
                and _keeps(s, level)
            ]
    return {
        "path": loaded.path,
        "language": result.language,
        "language_confidence": round(result.language_confidence, 3),
        "sentences": result.sentence_count if hasattr(result, "sentence_count") else len(result.sentences),
        "words": result.word_count,
        "pages": result.page_count,
        "blocks": len(result.blocks),
        "items": items[: max(1, max_items)],
        "truncated": len(items) > max(1, max_items),
        "next_step": (
            "checker_scan for AI-risk and plagiarism scoring, checker_similarity for "
            "the institutional percentages."
        ),
    }


def _keeps(sentence: Any, level: RiskLevel) -> bool:  # noqa: ANN401
    """Segmentation has no scores, so filtering falls back to length."""
    return len(sentence.text) > 0


@server.tool(
    name="checker_scan",
    title="Scan a thesis or document",
    description=(
        "Full analysis: per-sentence AI-authorship risk (two-model likelihood ratio, "
        "perplexity/burstiness, classifier, stylometry) plus plagiarism against the "
        "reference sources you pass. Returns a verdict, per-section breakdown and the "
        "top located findings; use checker_sentences / checker_matches to drill down. "
        "First run downloads model weights (~1 GB, slow). Prefer min_level='medium' and a "
        "small top on large theses."
    ),
    annotations=_ann(**_NEEDS_MODELS),
)
def checker_scan(
    path: str,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    ref_glob: str = "*",
    exclude: list[str] | None = None,
    language: str | None = None,
    profile: str = "document",
    min_level: str | None = "medium",
    top: int = 20,
    use_ratio: bool = True,
    use_perplexity: bool = True,
    use_classifier: bool = True,
    ppl_model: str | None = None,
    classifier_model: str | None = None,
) -> dict[str, Any]:
    """Sentence-level AI-risk and plagiarism scan."""
    report, _loaded, references, dropped, top_sentences = _run_scan(
        _settings(),
        path,
        refs=refs,
        ref_dirs=ref_dirs,
        ref_glob=ref_glob,
        exclude=exclude,
        language=language,
        profile=profile,
        min_level=min_level,
        top=top,
        use_ratio=use_ratio,
        use_perplexity=use_perplexity,
        use_classifier=use_classifier,
    )
    return {
        "document": _document_summary(report),
        "verdict": _verdict(report),
        "sections": _sections(report),
        "sources_used": [s["name"] for s in report.sources],
        "sources_skipped": dropped,
        "findings": [_sentence_payload(s) for s in top_sentences],
        "disclaimer": report.disclaimer,
        "next_step": (
            "checker_similarity for institutional percentages, checker_compliance for "
            "the YOK checklist, checker_sentences for the full sentence list."
        ),
    }


@server.tool(
    name="checker_similarity",
    title="Institutional similarity percentage",
    description=(
        "Turnitin-style similarity percentages with the filters Turkish thesis rules "
        "use: bibliography excluded, quotations included/excluded, matches under 5 words "
        "excluded, plus a per-source breakdown and the published thresholds of YTÜ, "
        "İstanbul, Başkent, ESÜ, Çukurova and Akdeniz. Also reports the citation status of "
        "every match (cited_and_quoted = legitimate reuse) and compares the percentage with "
        "the measured Turkish-thesis distribution (mean 28.7%, SD 11.58, n=600), the "
        "measured Turkish AI-presence distribution (204 articles, mean 20%), and - "
        "important for a thesis citing foreign literature - which languages the reference "
        "sources were in. Cross-language reuse detection is much weaker (precision 80% "
        "untranslated, 26.7% translated, 16.7% translated-then-paraphrased), so a zero "
        "with English-only references is not evidence of originality. No model "
        "weights needed."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_similarity(
    path: str,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    ref_glob: str = "*",
    exclude: list[str] | None = None,
    language: str | None = None,
    exclude_quotes: bool = False,
    keep_references: bool = False,
    min_match_words: int = 5,
    per_source_limit: int = 20,
) -> dict[str, Any]:
    """Similarity percentages under the institutional filter set."""
    settings = _settings()
    settings.plagiarism.include_quotes = not exclude_quotes
    settings.plagiarism.exclude_references = not keep_references
    settings.plagiarism.institutional_min_match_words = max(1, min_match_words)
    settings.profile = "thesis"

    report, _loaded, references, dropped, _top = _run_scan(
        settings,
        path,
        refs=refs,
        ref_dirs=ref_dirs,
        ref_glob=ref_glob,
        exclude=exclude,
        language=language,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        use_code=False,
    )
    similarity = report.similarity
    if similarity is None:
        return {
            "error": "kaynak verilmedi; --ref veya --ref_dir gerekli",
            "path": report.path,
        }
    payload = similarity.to_dict()
    payload["per_source"] = payload["per_source"][: max(1, per_source_limit)]
    # Flatten the headline percentages: a caller should not need to know how
    # ``SimilarityReport.to_dict`` nests them.
    payload.update(payload.pop("similarity", {}))
    # Lives on the document report, not on SimilarityReport: it needs the token
    # stream and the detected source languages, which the percentage does not.
    payload["cross_lingual"] = (
        report.cross_lingual.to_dict() if report.cross_lingual else None
    )
    cross_lingual = payload.get("cross_lingual")
    if cross_lingual and cross_lingual.get("clusters_found"):
        counts = cross_lingual.get("confidence_counts", {})
        payload["cross_lingual_note"] = (
            f"Kelime eşleştirmesi bu aktarımı göremedi "
            f"(benzerlik {payload.get('incl_quotes_percent', 0)}%). "
            f"Çap kümeleri: {counts.get('high', 0)} yüksek, "
            f"{counts.get('medium', 0)} orta, {counts.get('review', 0)} gözden geçir. "
            "'review' seviyesi yalnızca sayılardan oluşan kısa kümelerdir ve alan "
            "içinde tekrar eden örneklem büyüklüklerinden kaynaklanabilir."
        )
    coverage = payload.get("language_coverage")
    if coverage and coverage.get("cross_language_sources"):
        # Say it here rather than letting a zero be read as "no overlap": with
        # cross-language references, precision falls from 80% to 26.7%.
        payload["interpretation"] = (
            f"{coverage['reading']} A match found in a cross-language source is "
            "strong evidence; a match *not* found there is not evidence of absence."
        )
    return {
        "document": _document_summary(report),
        "similarity": payload,
        "sources_skipped": dropped,
        "sources_used": [s["name"] for s in report.sources],
        "disclaimer": report.disclaimer,
        "next_step": "checker_compliance turns this into a checklist.",
    }


@server.tool(
    name="checker_compliance",
    title="Thesis compliance checklist",
    description=(
        "YOK/institutional checklist for a thesis: is AI assistance disclosed, which "
        "matches lack attribution, which critical sections (results/discussion/conclusion/"
        "method) carry high AI scores, whether the similarity exceeds institutional "
        "thresholds, and how much of the document the AI signals could judge at all. "
        "Every item cites the rule it comes from. No model weights needed for the "
        "compliance items; pass use_ratio=false for a fast run."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_compliance(
    path: str,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    exclude: list[str] | None = None,
    language: str | None = None,
    use_ratio: bool = False,
    max_uncited: int = 15,
) -> dict[str, Any]:
    """Compliance checklist with evidence and rules."""
    settings = _settings()
    settings.profile = "thesis"
    report, loaded, _references, dropped, _top = _run_scan(
        settings,
        path,
        refs=refs,
        ref_dirs=ref_dirs,
        exclude=exclude,
        language=language,
        use_ratio=use_ratio,
        use_perplexity=use_ratio,
        use_classifier=False,
    )
    result = build_compliance_report(report, raw_text=loaded.text)
    payload = dict(result.to_dict())
    uncited = payload.get("uncited_matches")
    if isinstance(uncited, list):
        payload["uncited_matches"] = uncited[: max(1, max_uncited)]
    return {
        "document": _document_summary(report),
        "verdict": _verdict(report),
        "compliance": payload,
        "sources_skipped": dropped,
        "disclaimer": report.disclaimer,
    }


@server.tool(
    name="checker_sentences",
    title="List analysed sentences",
    description=(
        "Return analysed sentences with their scores, section role and reasons, so a "
        "finding can be traced back to the exact sentence. Filter by minimum risk level. "
        "Use it after checker_scan to page through what the summary listed, or to check "
        "the short sentences the AI signals deliberately skipped (below 12 words)."
    ),
    annotations=_ann(**_NEEDS_MODELS),
)
def checker_sentences(
    path: str,
    min_level: str | None = None,
    limit: int = 40,
    offset: int = 0,
    section: str | None = None,
    language: str | None = None,
    profile: str = "document",
    use_ratio: bool = True,
    use_perplexity: bool = True,
    use_classifier: bool = True,
) -> dict[str, Any]:
    """Paged list of scored sentences."""
    report, _loaded, _refs, _dropped, _top = _run_scan(
        _settings(),
        path,
        language=language,
        profile=profile,
        min_level=None,
        top=1,
        use_ratio=use_ratio,
        use_perplexity=use_perplexity,
        use_classifier=use_classifier,
    )
    rows = [
        s
        for s in report.sentences
        if (min_level is None or _level(min_level) is None
            or s.risk.at_least(_level(min_level) or RiskLevel.NONE))
        and (section is None or " › ".join(s.location.section_path) == section)
    ]
    window = rows[offset : offset + max(1, limit)]
    return {
        "document": _document_summary(report),
        "total_matching": len(rows),
        "offset": offset,
        "returned": len(window),
        "sentences": [_sentence_payload(s) for s in window],
        "next_step": "checker_matches for the plagiarism side of a sentence.",
    }


@server.tool(
    name="checker_matches",
    title="List plagiarism matches",
    description=(
        "Return plagiarism matches with offsets on both sides, the citation status of "
        "each match (cited_and_quoted / missing_citation / missing_quotation / "
        "not_cited_or_quoted) and the matched text. Filter to unattributed matches when "
        "the question is 'what needs a citation?'."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_matches(
    path: str,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    exclude: list[str] | None = None,
    language: str | None = None,
    unattributed_only: bool = True,
    min_words: int = 4,
    limit: int = 40,
) -> dict[str, Any]:
    """Located matches with attribution status."""
    from checker_app.domain.enums import CitationStatus  # noqa: PLC0415

    settings = _settings()
    settings.profile = "thesis"
    report, _loaded, references, dropped, _top = _run_scan(
        settings,
        path,
        refs=refs,
        ref_dirs=ref_dirs,
        exclude=exclude,
        language=language,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        use_code=False,
    )
    if not references:
        return {"error": "kaynak verilmedi; refs veya ref_dirs gerekli", "path": report.path}

    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for sentence in report.sentences:
        if not sentence.plagiarism:
            continue
        for match in sentence.plagiarism.matches:
            status = (
                match.citation_status.value
                if match.citation_status is not None
                else "unknown"
            )
            counts[status] = counts.get(status, 0) + 1
            if unattributed_only and match.citation_status in (
                CitationStatus.CITED_AND_QUOTED,
                None,
            ):
                continue
            if match.matched_words < min_words:
                continue
            items.append(
                {
                    "sentence": sentence.index,
                    "location": sentence.location.label(),
                    "section": " › ".join(sentence.location.section_path) or "(giriş)",
                    "kind": match.kind.value,
                    "source": match.source_name,
                    "matched_words": match.matched_words,
                    "character_range": [match.doc_char_start, match.doc_char_end],
                    "source_range": [match.source_char_start, match.source_char_end],
                    "citation_status": status,
                    "text": match.snippet[:220],
                    "context": match.context[:160],
                    "source_context": match.source_context[:160],
                }
            )
    items.sort(key=lambda row: -row["matched_words"])
    return {
        "document": _document_summary(report),
        "sources_used": [s["name"] for s in report.sources],
        "sources_skipped": dropped,
        "attribution_counts": counts,
        "total_matching": len(items),
        "matches": items[: max(1, limit)],
        "disclaimer": report.disclaimer,
    }


@server.tool(
    name="checker_compare_revisions",
    title="Compare two revisions",
    description=(
        "Sentence-level diff of two versions of the same text. Useful for revision work "
        "and as a self-plagiarism reference: the left revision is a legitimate source, so "
        "checker_matches with the left file as a reference shows what was carried over."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_compare_revisions(
    left_path: str,
    right_path: str,
    limit: int = 40,
    language: str | None = None,
) -> dict[str, Any]:
    """Sentence-level diff with locations."""
    from checker_app.domain.enums import DiffOp  # noqa: PLC0415

    settings = _settings()
    loader = SourceLoader(settings)
    left = loader.load_document(left_path)
    right = loader.load_document(right_path)
    diffs = ScanRunner(settings).diff(left.text, right.text)

    counts = {op.value: 0 for op in DiffOp}
    items: list[dict[str, Any]] = []
    for diff in diffs:
        counts[diff.op.value] += 1
        if diff.op is DiffOp.EQUAL:
            continue
        location = diff.left_location or diff.right_location
        items.append(
            {
                "op": diff.op.value,
                "left_index": diff.left_index,
                "right_index": diff.right_index,
                "location": location.label() if location else None,
                "left_text": (diff.left_text or "")[:200] or None,
                "right_text": (diff.right_text or "")[:200] or None,
            }
        )
    _ = language
    return {
        "left": left.path,
        "right": right.path,
        "counts": counts,
        "changed": items[: max(1, limit)],
        "total_changed": len(items),
    }


@server.tool(
    name="checker_calibrate",
    title="Calibrate on a local corpus",
    description=(
        "Measure the operating point of the likelihood-ratio signal on YOUR human "
        "writing, optionally against known AI text. Reports medians, p10-p90, AUC, the "
        "false-positive rate at the current threshold, whether the signal's direction "
        "is inverted for the loaded model pair (measured AUC 0.25 on one pair, 0.94 on "
        "another), and - when both classes are present - how far the score is from a "
        "probability (ECE, Brier, reliability bins, fitted temperature). The last part "
        "matters because calibration is separate from accuracy: dropping temperature "
        "scaling leaves F1 at 80.17 -> 80.16 while ECE goes 0.06 -> 0.12 "
        "(arXiv:2510.00890), and the threshold alone swings F1 by up to 0.8 across "
        "corpora for tau in [0.1, 0.5] (arXiv:2606.04906). Slow: loads two models. Do "
        "this before trusting any threshold on a new corpus."
    ),
    annotations=_ann(**_NEEDS_MODELS),
)
def checker_calibrate(
    human_dir: str,
    machine_dir: str | None = None,
    language: str | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Measure the signal on a local corpus."""
    settings = _settings()
    extensions = (".txt", ".md", ".markdown", ".rst")
    human = sorted(p for p in Path(human_dir).rglob("*") if p.suffix.lower() in extensions)
    machine = (
        sorted(p for p in Path(machine_dir).rglob("*") if p.suffix.lower() in extensions)
        if machine_dir
        else []
    )
    if not human:
        return {"error": f"metin dosyası yok: {human_dir}"}
    result = calibrate_corpus(
        settings,
        human_files=human,
        machine_files=machine,
        language=None if language in (None, "auto") else language,
    )
    payload = result.to_dict()
    if out_path:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["written_to"] = str(target)
    payload["files"] = len(human) + len(machine)
    return payload


@server.tool(
    name="checker_doctor",
    title="Check the environment",
    description=(
        "Report which models and features are available on this machine, whether the "
        "classifier is usable for the detected language, and which signals would be "
        "skipped. Call it first when a scan looks unexpectedly clean or empty."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_doctor(language: str | None = None) -> dict[str, Any]:
    """Environment and signal availability."""
    import importlib  # noqa: PLC0415

    settings = _settings()
    lang = language or "tr"

    features: dict[str, bool] = {}
    for module in ("torch", "transformers", "bs4", "pypdf", "httpx"):
        try:
            importlib.import_module(module)
            features[module] = True
        except ImportError:
            features[module] = False

    from checker_app.services.ratio import observer_models  # noqa: PLC0415

    notes: list[str] = []
    classifier_usable = True
    if lang != "en" and not settings.ai.classifier_model.lower().startswith(("mult", "xlm")):
        classifier_usable = False
        notes.append(
            f"{settings.ai.classifier_model} yalnızca İngilizce eğitilmiştir; "
            f"{lang} metinde otomatik olarak düşürülür."
        )
    if not features.get("torch") or not features.get("transformers"):
        notes.append(
            "torch/transformers yok: perplexity, likelihood-ratio ve sınıflandırıcı "
            "çalışmaz; yalnız intihal ve stilometri çalışır."
        )
    observer, performer = observer_models(
        settings.language_model(lang), settings.ratio_model(lang)
    )
    return {
        "python_features": features,
        "language": lang,
        "models": {
            "perplexity": settings.language_model(lang),
            "likelihood_ratio": f"{observer} | {performer}",
            "classifier": settings.ai.classifier_model,
            "classifier_usable": classifier_usable,
        },
        "weights": {
            "perplexity": settings.scoring.weight_perplexity,
            "likelihood_ratio": settings.scoring.weight_ratio,
            "classifier": settings.scoring.weight_classifier,
            "stylometry": settings.scoring.weight_stylometry,
            "uniformity": settings.scoring.weight_uniformity,
        },
        "operating_point": {
            "language": lang,
            "ratio_center": settings.ratio_threshold(lang),
            "ratio_scale": settings.scoring.ratio_scale,
            "fpr_budget": settings.scoring.fpr_budget,
            "min_ai_words": settings.ai.min_ai_words,
        },
        "notes": notes,
        "next_step": "checker_scan with use_ratio=false for a fast offline pass.",
    }


@server.tool(
    name="checker_patchwork",
    title="Patchwriting shape of the reuse",
    description=(
        "How the reuse is *assembled*, not how much of it there is: greedily accepted "
        "runs of >= 5 reused words, normalised by document words (HyPlag's Greedy "
        "Identifier Tiles, JCDL 2019). Reports the score against the one chance baseline "
        "in this literature that was actually measured (1,000,000 random document pairs: "
        "GIT >= 0.15), the mean and largest tile size, and the spread-vs-concentrated "
        "reading. COPE's concentration principle means many short runs are *less* "
        "concerning than a few long blocks, so both numbers are returned. No model "
        "weights needed."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_patchwork(
    path: str,
    refs: list[str] | None = None,
    ref_dirs: list[str] | None = None,
    ref_glob: str = "*",
    exclude: list[str] | None = None,
    language: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Quantity-sensitive reuse shape with its chance baseline."""
    settings = _settings()
    settings.profile = "thesis"
    report, _loaded, references, dropped, _top = _run_scan(
        settings,
        path,
        refs=refs,
        ref_dirs=ref_dirs,
        ref_glob=ref_glob,
        exclude=exclude,
        language=language,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        use_code=False,
    )
    if not references:
        return {"error": "kaynak verilmedi; refs veya ref_dirs gerekli", "path": report.path}
    if report.patchwork is None:
        return {"error": "yamalama raporu üretilemedi", "path": report.path}
    payload = report.patchwork.to_dict()
    payload["tiles"] = payload["tiles"][: max(1, limit)]
    return {
        "document": _document_summary(report),
        "patchwork": payload,
        "chance_baselines_measured": CHANCE_BASELINES,
        "sources_skipped": dropped,
        "disclaimer": report.disclaimer,
    }


@server.tool(
    name="checker_references",
    title="Audit the reference list",
    description=(
        "Structural plausibility of every bibliography entry: persistent identifier "
        "(DOI/arXiv/URL), year plausibility, author shape, venue, and malformed IDs. "
        "Flags only clusters, never a single missing field, because a print-only "
        "Turkish reference legitimately has no DOI. Motivation, with numbers: The Lancet "
        "(2026) found 4,046 fabricated references in 2,810 papers and the rate rose "
        ">12x in three years (1 in 2,828 in 2023 -> 1 in 277 in early 2026); 91% of "
        "affected papers carry only 1-2 fabricated entries. Adelphi's 2025 guidance lists "
        "hallucinated references as the corroborating evidence a faculty report needs "
        "besides a detector score. STRUCTURAL ONLY - it does not resolve DOIs or query "
        "CrossRef, so it can flag what to look up by hand but cannot prove absence."
    ),
    annotations=_ann(**_READ_ONLY_LOCAL),
)
def checker_references(
    path: str,
    show_all: bool = False,
    limit: int = 40,
) -> dict[str, Any]:
    """Reference-list plausibility audit."""
    settings = _settings()
    settings.profile = "thesis"
    report, loaded, _refs, _dropped, _top = _run_scan(
        settings,
        path,
        min_level=None,
        top=1,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        use_code=False,
    )
    audit = report.references
    if audit is None or not audit.total:
        return {
            "error": "kaynakça bölümü bulunamadı veya boş",
            "path": report.path,
        }
    payload = audit.to_dict(limit=limit, only_flagged=not show_all)
    _ = loaded
    return {
        "document": _document_summary(report),
        "references": payload,
        "next_step": "Her 'review' kaydı için DOI'yi CrossRef'te elle arayın.",
    }


@server.resource(
    "checker://disclaimer",
    name="checker-disclaimer",
    title="What this tool may and may not be used for",
    mime_type="text/plain",
)
async def _disclaimer_resource() -> str:
    from checker_app.services.scoring import DISCLAIMER  # noqa: PLC0415

    return (
        f"{DISCLAIMER}\n\n"
        f"{AI_DEGREE_CAVEAT}\n\n"
        f"{AI_HUMAN_BASELINE}\n\n"
        f"{AI_TURKISH_FPR}\n\n"
        "YÖK, ulusal bir benzerlik yüzdesi belirlemiyor; her enstitü kurulu belirliyor. "
        "Yaygın Türkçe filtreler: kaynakça hariç, alıntılar dahil, 5 kelimeden az "
        "eşleşmeler hariç.\n\n"
        "İki dilin tabanı yaklaşık 3 KAT farklıdır: Türkçe tezler %28.7 ± 11.58 "
        "(n=600), temiz İngilizce doktora tezleri %9 ± 6 (n=360). İngilizce için "
        "ÖLÇÜLMÜŞ optimal eşik %15'tir (duyarlılık %84.8, özgüllük %80.5, AUC "
        "0.902); %25-30 bir politik tavandır, gözlenen normal değil. "
        "Yayımlanmış İngilizce AI yaygınlığı alana göre değişir: arXiv CS özetleri "
        "%22.5, matematik %7.7 (girişler %4.1), tahmin hatası <3.5 puan.\n\n"
        "Kurumsal arka plan: RAID (arXiv:2405.07940) ekibi dedektörlerin cezai "
        "bağlamda kullanımına itiraz ediyor; IEEE S&P 2026 ticari dedektörlerin "
        "akademik kararlarda kullanım için uygun olmadığını ve FPR aralığının "
        "%0.05-68.6 olduğunu bildiriyor (N=6.295). Vanderbilt (2023), UCLA ve "
        "UC San Diego (2024), Waterloo (Eylül 2025) ve Curtin (Ocak 2026) Turnitin "
        "AI dedektörünü tamamen devre dışı bıraktı.\n\n"
        "Raporlama tabanı: %20'nin altındaki bir AI kapsamı için belge düzeyinde hüküm "
        "verilmez, yalnız cümle düzeyi bulgular verilir (Turnitin AIW-2, Ağu. 2024: "
        "719.877 insan öğrenci metninde belge FPR %0.51, cümle FPR %0.33).\n\n"
        "Yapı profili yönü: insan yazarlar "
        f"{RELATION_DIRECTION['attribution']}/{RELATION_DIRECTION['temporal']} ilişkilerini, "
        f"LLM'ler {RELATION_DIRECTION['elaboration']}/{RELATION_DIRECTION['cause']} "
        "ilişkilerini fazla kullanır (arXiv:2604.04932); bu yön düzenlemeden sonra da "
        "korunur (insan→AI cilalama kosinüs benzerliği 0.92±0.08).\n\n"
        "Ayrıntılı kanıt listesi: checker/LITERATURE.md"
    )


# ---------------------------------------------------------------------------- entry
def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8090) -> None:
    """Entry point for ``checker mcp``."""
    if transport == "stdio":
        asyncio.run(server.run_stdio_async())
    elif transport == "http":
        server.run(transport="streamable-http", host=host, port=port)
    else:  # pragma: no cover - the CLI validates this
        raise ValueError(f"unknown transport {transport!r}")


def main() -> None:  # pragma: no cover - process entry point
    """``python -m checker_app.mcp.server`` - stdio by default."""
    run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()


__all__ = ["server", "run", "main", "INSTRUCTIONS"]