"""Orchestration: text in, :class:`DocumentReport` out.

The runner is the only place that knows the order of operations and how a
failure of one signal is handled:

1. segment the document into located sentences
2. run every signal, each in its own ``try`` so a missing model degrades the
   report instead of failing the scan
3. credit matches to sentences and fuse everything into scores
4. package the result with the settings and models that produced it

Signals that could not run are listed in ``degraded_signals``; the CLI prints
them so a clean report is never mistaken for a full one.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

from checker_app.config import CheckerSettings
from checker_app.domain.enums import CitationStatus, DiffOp
from checker_app.domain.models import (
    DocumentReport,
    MatchSpan,
    Sentence,
    SentenceDiff,
    SourceDocument,
)
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.domain.text import language_confidence
from checker_app.logging import get_logger
from checker_app.services.attribution import (
    classify_attribution,
    find_citation_spans,
    find_quotation_spans,
)
from checker_app.services.code_ast import CodeBlock, CodeCompareService
from checker_app.services.detector import DetectorService
from checker_app.services.discourse import discourse_profile
from checker_app.services.integrity import check_integrity
from checker_app.services.patchwork import PatchworkReport, build_patchwork_report
from checker_app.services.perplexity import PerplexityService, SignalUnavailable
from checker_app.services.plagiarism import PlagiarismService, drop_shadowed
from checker_app.services.ratio import LikelihoodRatioService, observer_models
from checker_app.services.references import audit_references
from checker_app.services.scoring import (
    AI_DEGREE_CAVEAT,
    AI_HUMAN_BASELINE,
    DISCLAIMER,
    Scorer,
    is_prose,
)
from checker_app.services.similarity import SimilarityReport, build_similarity_report
from checker_app.services.sources import LoadedDocument, SourceLoader
from checker_app.services.stylometry import StylometryService

__all__ = ["ScanRequest", "ScanRunner"]

logger = get_logger("checker.runner")


@dataclass(slots=True)
class ScanRequest:
    """One scan: a document plus its references and the signals to run."""

    text: str
    path: str = "<stdin>"
    references: Sequence[LoadedDocument] = ()
    language: str | None = None
    use_perplexity: bool = True
    use_classifier: bool = True
    use_ratio: bool = True
    use_plagiarism: bool = True
    use_code: bool = True
    perplexity_model: str | None = None
    ratio_model: str | None = None
    classifier_model: str | None = None


def reference_token_indices(sentences: Sequence[Sentence]) -> set[int]:
    """Tokens inside sections that are excluded from overlap by institutional rule.

    "Kaynakca hariç" is in every Turkish thesis rule sheet, and comparing a
    reference list against the sources it names is a guaranteed 100% "match"
    that means nothing (arXiv:2609.32963 strips the same material).
    """
    out: set[int] = set()
    for sentence in sentences:
        # ``matched`` True means "this section takes part in comparison"; the
        # excluded ones are the reference list, acknowledgements and front
        # matter.
        if not sentence.location.section_role.matched:
            out.update(range(sentence.token_start, sentence.token_end))
    return out


class ScanRunner:
    """Run the whole pipeline for one document."""

    def __init__(
        self,
        settings: CheckerSettings,
        *,
        sources: Sequence[SourceDocument] = (),
        ratio_service: LikelihoodRatioService | None = None,
    ) -> None:
        self.settings = settings
        self.sources = list(sources)
        self.loader = SourceLoader(settings)
        self.splitter = SentenceSplitter(settings.segmentation)
        self.stylometry = StylometryService(settings.stylometry)
        self.plagiarism = PlagiarismService(settings.plagiarism)
        self.code = CodeCompareService(settings.plagiarism)
        # Injectable so the test suite exercises the wiring without weights.
        self._ratio_service = ratio_service

    # ------------------------------------------------------------------ public
    def scan(self, request: ScanRequest) -> DocumentReport:
        started = time.perf_counter()
        text = request.text[: self.settings.max_chars]
        degraded: list[str] = []
        models: dict[str, str] = {}

        segmentation = self.splitter.split(text)
        sentences = segmentation.sentences
        language = self._language(
            request, text, segmentation.language, segmentation.language_confidence
        )

        # Byte-level health first: a homoglyph or zero-width-character attack
        # invalidates every surprisal-based score (RAID: 40.6% average accuracy
        # loss under homoglyph), so it has to be known before the scores are read.
        integrity = check_integrity(text)
        tamper = integrity.as_degradation()
        if tamper:
            degraded.append(f"text_integrity: {tamper}")

        # Discourse profile over the judged prose. Lexicon-only, no model, and
        # the one structural signal whose measured direction survives editing
        # (RACE, arXiv:2604.04932: human -> AI-polished cosine 0.92 +/- 0.08).
        discourse = discourse_profile(text, segmentation.tokens)

        # Reference-list plausibility. Free, offline, and the corroborating
        # evidence Adelphi's 2025 guidance asks for alongside a detector score.
        reference_audit = audit_references(sentences)

        # Headings, code, math and tables are located and reported but never
        # judged for authorship: a model score on "# Giriş" is noise, and it
        # would poison the document averages and the z-scores.
        judged = [s for s in sentences if is_prose(s) and s.location.section_role.scored]

        styles = {
            sentence.index: self.stylometry.analyze(
                sentence, segmentation.tokens[sentence.token_start : sentence.token_end]
            )
            for sentence in sentences
        }

        # Document-level context that is deliberately *not* scored: the lexical
        # richness trio (strongest feature area in the literature, unstable sign)
        # and the length confound (word count alone reaches ROC-AUC 0.68).
        style_stats = self.stylometry.document_stats(sentences, segmentation.tokens)

        perplexities: dict = {}
        if request.use_perplexity and judged:
            model = request.perplexity_model or self.settings.language_model(language)
            models["perplexity"] = model
            perplexity = PerplexityService(
                self.settings.ai, model_id=model, batch_size=self.settings.ai.batch_size
            )
            try:
                perplexities = perplexity.score(judged)
            except SignalUnavailable as exc:
                degraded.append(f"perplexity: {exc}")
                logger.warning("perplexity çalışmadı: %s", exc)
            except Exception as exc:  # pragma: no cover - model/runtime failures
                degraded.append(f"perplexity: {type(exc).__name__}: {exc}")
                logger.exception("perplexity hatası")

        ratios: dict = {}
        if request.use_ratio and judged:
            observer, performer = observer_models(
                request.perplexity_model or self.settings.language_model(language),
                request.ratio_model or self.settings.ratio_model(language),
            )
            models["likelihood_ratio"] = f"{observer} | {performer}"
            ratio_service = self._ratio_service or LikelihoodRatioService(
                self.settings.ai, observer_model=observer, performer_model=performer
            )
            try:
                ratios = ratio_service.score(judged)
            except SignalUnavailable as exc:
                degraded.append(f"likelihood_ratio: {exc}")
                logger.warning("likelihood-ratio çalışmadı: %s", exc)
            except Exception as exc:  # pragma: no cover - model/runtime failures
                degraded.append(f"likelihood_ratio: {type(exc).__name__}: {exc}")
                logger.exception("likelihood-ratio hatası")

        classifiers: dict = {}
        if request.use_classifier and judged:
            model = request.classifier_model or self.settings.ai.classifier_model
            models["classifier"] = model
            detector = DetectorService(self.settings.ai, model_id=model)
            try:
                classifiers = detector.score(judged)
            except SignalUnavailable as exc:
                degraded.append(f"classifier: {exc}")
                logger.warning("sınıflandırıcı çalışmadı: %s", exc)
            except Exception as exc:  # pragma: no cover - model/runtime failures
                degraded.append(f"classifier: {type(exc).__name__}: {exc}")
                logger.exception("sınıflandırıcı hatası")

        sources = list(self.sources)
        plagiarism_stats: dict = {}
        matches: list = []
        similarity: SimilarityReport | None = None
        patchwork: PatchworkReport | None = None
        if request.use_plagiarism and sources and segmentation.tokens:
            result = self.plagiarism.compare(sentences, segmentation.tokens, sources, text)
            matches.extend(result.matches)
            plagiarism_stats.update(result.sentences)
            degraded.extend(result.degraded)
            if request.use_code and self.settings.plagiarism.check_code:
                source_blocks: dict[str, Sequence[CodeBlock]] = {
                    source.source_id: self.code.extract_blocks(source.text, source.tokens, language)
                    for source in sources
                }
                try:
                    code_matches = self.code.compare(
                        sentences, text, segmentation.tokens, sources, source_blocks
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("kod karşılaştırma hatası")
                    degraded.append(f"code_ast: {type(exc).__name__}: {exc}")
                else:
                    matches.extend(code_matches)
            matches = drop_shadowed(matches)

            # Attribution: quoted+cited passages are legitimate reuse. Only 6.4%
            # of matched pairs in a 2025-26 corpus study carried a citation at
            # all (arXiv:2609.32963), so an unweighted match count would flag
            # correct academic writing as misconduct.
            quote_spans = find_quotation_spans(text)
            citation_spans = find_citation_spans(text)
            attributed: list[MatchSpan] = []
            citation_status: dict[int, CitationStatus] = {}
            for match in matches:
                evidence = classify_attribution(
                    text,
                    match.doc_char_start,
                    match.doc_char_end,
                    quote_spans=quote_spans,
                    citation_spans=citation_spans,
                )
                attributed_match = replace(
                    match,
                    citation_status=evidence.status,
                    citation_evidence=(evidence.citation_evidence or evidence.quote_evidence),
                )
                attributed.append(attributed_match)
                citation_status[id(attributed_match)] = evidence.status
            matches = attributed
            plagiarism_stats = self.plagiarism.credit(sentences, matches)

            similarity = build_similarity_report(
                tokens=segmentation.tokens,
                matches=matches,
                total_words=segmentation.word_count,
                reference_token_indices=reference_token_indices(sentences),
                quote_spans=quote_spans,
                statuses=citation_status,
                min_match_words=self.settings.plagiarism.institutional_min_match_words,
                exclude_references=self.settings.plagiarism.exclude_references,
                include_quotes=self.settings.plagiarism.include_quotes,
            )

            # Quantity-sensitive companion to the percentage: how many distinct
            # reused runs the document is assembled from, and whether that beats
            # the measured chance baseline (HyPlag GIT >= 0.15, arXiv:1906.11761).
            patchwork = build_patchwork_report(
                tokens=segmentation.tokens,
                matches=matches,
                total_words=segmentation.word_count,
                quote_spans=quote_spans,
                statuses=citation_status,
            )

        scorer = Scorer(self.settings, language)
        scores = scorer.score(
            sentences, styles, perplexities, classifiers, plagiarism_stats, degraded, ratios
        )

        return DocumentReport(
            path=request.path,
            language=language,
            language_confidence=(
                segmentation.language_confidence if request.language is None else 1.0
            ),
            char_count=segmentation.char_count,
            word_count=segmentation.word_count,
            sentence_count=len(sentences),
            unscored_sentences=sum(1 for s in scores.sentences if s.unscored_for_ai),
            ai_share=scores.ai_share,
            ai_flagged_words=scores.ai_flagged_words,
            below_ai_reporting_floor=scores.below_ai_reporting_floor,
            page_count=segmentation.page_count,
            sentences=scores.sentences,
            sections=scores.sections,
            ai_score=scores.ai_score,
            plagiarism_score=scores.plagiarism_score,
            risk_score=scores.risk_score,
            risk=scores.risk,
            flagged_sentences=scores.flagged,
            high_risk_sentences=scores.high_risk,
            matched_sentences=scores.matched,
            mean_perplexity=scores.mean_perplexity,
            perplexity_burstiness=scores.perplexity_burstiness,
            length_burstiness=scores.length_burstiness,
            models=models,
            degraded_signals=tuple(dict.fromkeys(scores.degraded)),
            similarity=similarity,
            patchwork=patchwork,
            discourse=discourse,
            style_stats=style_stats,
            references=reference_audit,
            integrity=integrity,
            sources=tuple(
                {
                    "id": source.source_id,
                    "name": source.name,
                    "kind": source.kind,
                    "chars": len(source.text),
                    "words": len(source.tokens),
                }
                for source in sources
            ),
            settings=self._settings_snapshot(request),
            duration_seconds=round(time.perf_counter() - started, 3),
            disclaimer=DISCLAIMER,
            degree_caveat=AI_DEGREE_CAVEAT,
            human_baseline=AI_HUMAN_BASELINE,
        )

    def diff(self, left_text: str, right_text: str) -> tuple[SentenceDiff, ...]:
        """Sentence-level diff of two revisions.

        Useful on its own ("what changed between v1 and v2") and as a self
        plagiarism reference: the left revision is a legitimate source.
        """
        left = self.splitter.split(left_text).sentences
        right = self.splitter.split(right_text).sentences
        matcher = difflib.SequenceMatcher(
            None, [s.text for s in left], [s.text for s in right], autojunk=False
        )
        diffs: list[SentenceDiff] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    sentence = right[j1 + offset]
                    diffs.append(
                        SentenceDiff(
                            op=DiffOp.EQUAL,
                            left_index=left[i1 + offset].index,
                            right_index=sentence.index,
                            left_text=left[i1 + offset].text,
                            right_text=sentence.text,
                            similarity=1.0,
                            left_location=left[i1 + offset].location,
                            right_location=sentence.location,
                        )
                    )
                continue
            op = {
                "insert": DiffOp.INSERT,
                "delete": DiffOp.DELETE,
                "replace": DiffOp.REPLACE,
            }[tag]
            for index in range(i1, i2):
                diffs.append(
                    SentenceDiff(
                        op=op,
                        left_index=left[index].index,
                        right_index=None,
                        left_text=left[index].text,
                        right_text=None,
                        similarity=0.0,
                        left_location=left[index].location,
                        right_location=None,
                    )
                )
            for index in range(j1, j2):
                diffs.append(
                    SentenceDiff(
                        op=op,
                        left_index=None,
                        right_index=right[index].index,
                        left_text=None,
                        right_text=right[index].text,
                        similarity=0.0,
                        left_location=None,
                        right_location=right[index].location,
                    )
                )
        return tuple(diffs)

    # ----------------------------------------------------------------- private
    def _language(self, request: ScanRequest, text: str, detected: str, confidence: float) -> str:
        configured = request.language or self.settings.language
        if configured in {"tr", "en"}:
            return configured
        if configured == "auto" or not request.language:
            if detected in {"tr", "en"}:
                return detected
            return language_confidence(text[:4000])[0] if text else "en"
        return "en"

    def _settings_snapshot(self, request: ScanRequest) -> dict[str, object]:
        scoring = self.settings.scoring
        return {
            "language": request.language or self.settings.language,
            "ngram_size": self.settings.plagiarism.ngram_size,
            "min_match_words": self.settings.plagiarism.min_match_words,
            "near_match": self.settings.plagiarism.near_match,
            "context_sentences": self.settings.ai.context_sentences,
            "weights": {
                "perplexity": scoring.weight_perplexity,
                "classifier": scoring.weight_classifier,
                "stylometry": scoring.weight_stylometry,
                "uniformity": scoring.weight_uniformity,
                "plagiarism": scoring.plagiarism_weight,
            },
            "risk_thresholds": dict(scoring.risk_thresholds),
        }
