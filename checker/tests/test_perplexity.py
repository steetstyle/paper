"""Perplexity service: windowing, context and degradation - no weights needed."""

from __future__ import annotations

import math

import pytest
from conftest import FakeLM

from checker_app.config import AIModelSettings
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.perplexity import (
    LMRequest,
    NullBackend,
    PerplexityService,
    SignalUnavailable,
    burstiness,
    cv,
)

TEXT = "Birinci cümle burada bulunmaktadır. İkinci cümle de oldukça uzundur. Üçüncü cümle geldi."


def build(nll: float = 3.0, context: int = 2) -> tuple[PerplexityService, FakeLM, object]:
    settings = AIModelSettings(context_sentences=context)
    backend = FakeLM(nll=nll)
    service = PerplexityService(settings, backend=backend, model_id="fake", batch_size=2)
    result = SentenceSplitter().split(TEXT)
    return service, backend, result


def test_a_sentence_is_never_its_own_context() -> None:
    """Regression: slicing by document index made ppl ~1 for everything."""
    service, backend, result = build()
    service.score(result.sentences)

    for sentence, request in zip(result.sentences, backend.requests, strict=True):
        body = request.text[request.score_from_char :]
        assert body == sentence.text, "scored span must be the sentence alone"
        prefix = request.text[: request.score_from_char].strip()
        assert sentence.text not in prefix or prefix == sentence.text[: len(prefix)]


def test_context_uses_list_position_not_document_index() -> None:
    settings = AIModelSettings(context_sentences=2)
    backend = FakeLM()
    service = PerplexityService(settings, backend=backend, model_id="fake")
    result = SentenceSplitter().split("# Başlık\n\nBirinci cümle burada. İkinci cümle burada.")

    judged = [s for s in result.sentences if s.index != 0]  # heading filtered out
    service.score(judged)

    first = backend.requests[0]
    assert first.score_from_char == 0, "no earlier sentence exists in the filtered list"
    assert backend.requests[1].text.startswith("Birinci cümle burada.")


def test_zero_context_scores_sentences_in_isolation() -> None:
    service, backend, result = build(context=0)
    service.score(result.sentences)
    assert all(request.score_from_char == 0 for request in backend.requests)


def test_stats_are_derived_from_nll() -> None:
    service, _, result = build(nll=math.log(42.0))
    scores = service.score(result.sentences)
    for stats in scores.values():
        assert stats.perplexity == pytest.approx(42.0, rel=1e-6)
        assert stats.log10_perplexity == pytest.approx(math.log10(42.0), rel=1e-6)
        assert stats.model == "fake"


def test_missing_model_degrades_to_empty_not_exception() -> None:
    service = PerplexityService(
        AIModelSettings(use_perplexity=True), backend=NullBackend(), model_id="nope"
    )
    result = SentenceSplitter().split(TEXT)
    assert service.score(result.sentences) == {}


def test_disabled_setting_raises_signal_unavailable() -> None:
    service = PerplexityService(AIModelSettings(use_perplexity=False), model_id="nope")
    with pytest.raises(SignalUnavailable):
        service.load()


def test_request_boundaries() -> None:
    request = LMRequest(text="önceki cümle. şimdiki cümle.", score_from_char=14)
    assert request.text[request.score_from_char :] == "şimdiki cümle."


def test_burstiness_and_cv() -> None:
    assert burstiness([1.0]) == 0.0
    assert burstiness([1.0, 1.0]) == 0.0
    assert burstiness([1.0, 3.0]) == pytest.approx(1.0)
    assert cv([2.0, 2.0]) == 0.0
    assert cv([1.0, 3.0]) == pytest.approx(0.5)
