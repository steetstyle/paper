"""Classifier adapter: label resolution, context, and the useless-signal guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import FakeClassifier

from checker_app.config import AIModelSettings
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.detector import DetectorService, HFDetector, NullDetector
from checker_app.services.perplexity import SignalUnavailable

TEXT = "# Başlık\n\nBirinci cümle burada oldukça uzun. İkinci cümle de burada ve uzundur."


def test_ai_label_is_resolved_by_name_not_position() -> None:
    human_first = SimpleNamespace(id2label={0: "ChatGPT", 1: "Human"})
    human_last = SimpleNamespace(id2label={0: "Human", 1: "ChatGPT"})
    assert HFDetector._resolve_label_index(human_first, "ai") == 0
    assert HFDetector._resolve_label_index(human_last, "ai") == 1


def test_unknown_labels_fall_back_without_crashing() -> None:
    config = SimpleNamespace(id2label={0: "label_a", 1: "label_b"})
    assert HFDetector._resolve_label_index(config, "ai") == 1
    assert HFDetector._resolve_label_index(config, "human") == 0


def test_context_uses_position_after_filtering() -> None:
    settings = AIModelSettings(classifier_context_sentences=1)
    backend = FakeClassifier(probability=0.7)
    service = DetectorService(settings, backend=backend, model_id="fake")
    result = SentenceSplitter().split(TEXT)
    judged = [s for s in result.sentences if s.index != 0]

    stats = service.score(judged)

    assert judged[0].text in backend.texts[0], "context is symmetric (span=1)"
    assert backend.texts[0].endswith(judged[1].text)
    # span=1 gives the middle sentence its neighbour on both sides.
    assert judged[1].text in backend.texts[1]
    assert backend.texts[1].startswith(judged[0].text)
    assert all(s.p_ai == pytest.approx(0.7) for s in stats.values())


def test_flat_zero_signal_is_dropped_as_useless() -> None:
    """An English-only head on Turkish text says 0% everywhere: no information."""
    service = DetectorService(
        AIModelSettings(), backend=FakeClassifier(probability=0.0), model_id="fake"
    )
    with pytest.raises(SignalUnavailable, match="hiçbir cümlede"):
        service.score(SentenceSplitter().split(TEXT).sentences)


def test_saturated_signal_is_dropped_as_useless() -> None:
    service = DetectorService(
        AIModelSettings(), backend=FakeClassifier(probability=0.99), model_id="fake"
    )
    with pytest.raises(SignalUnavailable, match="doygun"):
        service.score(SentenceSplitter().split(TEXT).sentences)


def test_null_backend_returns_nothing() -> None:
    service = DetectorService(AIModelSettings(), backend=NullDetector(), model_id="none")
    assert service.score(SentenceSplitter().split(TEXT).sentences) == {}
