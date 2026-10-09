"""Shared fixtures.

No test downloads a model: the model-backed layers are exercised through fake
backends, so the suite runs in under a second on a laptop with no network.
"""

from __future__ import annotations

import pytest

from checker_app.config import CheckerSettings
from checker_app.domain.segmentation import SegmentationResult, SentenceSplitter
from checker_app.services.perplexity import LMRequest, LMScore

TR_TEXT = """# Giriş

Yapay zeka ile üretilen metinler genellikle düşük perplexity değerine sahiptir. Bu çalışmada yöntem detaylı bir şekilde incelenmiştir.

Ayrıca, öğrencilerin yazım süreçleri üzerine ayrıntılı bir analiz yapılmıştır. Sonuç olarak, bulgular umut vericidir.

## Yöntem

Veri temizliği gerçekleştirildi. Model eğitimi için standart bir protokol kullanılmıştır.
"""

EN_TEXT = """# Introduction

Large language models have reshaped academic writing. The reliability of detectors remains contested in the literature.

However, the practical answer is pedagogical rather than technical. That is where the discussion should focus.
"""


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path) -> None:
    """No test may download weights or grab the GPU.

    A test that reaches the network takes minutes and fails when the machine is
    busy; the model-backed layers are covered through injected fakes instead.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CHECKER_AI_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def settings() -> CheckerSettings:
    return CheckerSettings(
        language="auto",
        ai={"use_perplexity": True, "use_classifier": True, "batch_size": 2},
    )


@pytest.fixture
def splitter(settings: CheckerSettings) -> SentenceSplitter:
    return SentenceSplitter(settings.segmentation)


@pytest.fixture
def tr_result(splitter: SentenceSplitter) -> SegmentationResult:
    return splitter.split(TR_TEXT)


@pytest.fixture
def en_result(splitter: SentenceSplitter) -> SegmentationResult:
    return splitter.split(EN_TEXT)


class FakeLM:
    """Records the requests the service builds, so tests can assert on them."""

    name = "fake-lm"
    context_size = 1024

    def __init__(self, nll: float = 3.0, token_count: int = 12) -> None:
        self.nll = nll
        self.token_count = token_count
        self.requests: list[LMRequest] = []

    def score(self, requests: list[LMRequest]) -> list[LMScore]:
        self.requests.extend(requests)
        return [
            LMScore(
                nll=self.nll,
                token_count=self.token_count,
                prefix_tokens=len(self._ids(r)),
            )
            for r in requests
        ]

    @staticmethod
    def _ids(request: LMRequest) -> list[int]:
        return request.text[: request.score_from_char].split()


class FakeClassifier:
    """Returns a fixed probability, and records what text it was given."""

    name = "fake-clf"

    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability
        self.texts: list[str] = []

    def p_ai(self, texts: list[str]) -> list[float]:
        self.texts.extend(texts)
        return [self.probability for _ in texts]
