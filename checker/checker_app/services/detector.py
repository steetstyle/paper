"""Sequence-classification AI detector (HuggingFace, local inference).

``Hello-SimpleAI/chatgpt-detector-roberta`` and friends are RoBERTa heads with
``{"Human", "ChatGPT"}`` labels. Two things matter for a sentence-level report:

* label ids are resolved by **name**, not position: the two popular checkpoints
  disagree on which index is which
* the classifier has a 512 token window, so sentences are scored with a small
  amount of neighbouring context and long sentences are chunked, with the
  per-chunk probabilities averaged

The output is a probability, reported as ``p_ai``, and it is treated as one
signal among several. See the README for why a single detector cannot decide
authorship.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from checker_app.config import AIModelSettings
from checker_app.domain.models import ClassifierStats, Sentence
from checker_app.logging import get_logger
from checker_app.services.perplexity import SignalUnavailable

__all__ = ["ClassifierBackend", "HFDetector", "NullDetector", "DetectorService"]

logger = get_logger("checker.detector")

_AI_LABELS = frozenset({"chatgpt", "ai", "ai-generated", "machine", "fake", "gpt"})
_HUMAN_LABELS = frozenset({"human", "real", "manual", "human-written", "student", "authentic"})
# Below / above these bounds the classifier is not discriminating at all.
_USELESS_BELOW = 0.05
_USELESS_ABOVE = 0.95


class ClassifierBackend(Protocol):
    name: str

    def p_ai(self, texts: Sequence[str]) -> list[float]:
        """Probability that each text is machine generated."""
        ...


@dataclass(slots=True)
class NullDetector:
    name: str = "unavailable"

    def p_ai(self, texts: Sequence[str]) -> list[float]:
        return [0.0] * len(texts)


class HFDetector:
    """RoBERTa-style sequence classifier, loaded lazily."""

    def __init__(self, model_id: str, settings: AIModelSettings) -> None:
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise SignalUnavailable(
                "torch/transformers kurulu değil. `pip install 'checker-app[ai]'`."
            ) from exc

        from checker_app.services.model_fetch import ensure_model  # noqa: PLC0415

        resolved = ensure_model(model_id, settings.cache_dir) or model_id
        self._torch = torch
        self.name = model_id
        self._settings = settings
        self._tokenizer = AutoTokenizer.from_pretrained(resolved)
        kwargs: dict[str, object] = {}
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device != "cpu":
            from checker_app.services.perplexity import HFCausalLM  # noqa: PLC0415

            kwargs[HFCausalLM._dtype_arg()] = torch.float16
        self._model = AutoModelForSequenceClassification.from_pretrained(resolved, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._ai_index = self._resolve_label_index(self._model.config, "ai")

    # ------------------------------------------------------------------ public
    def p_ai(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        torch = self._torch
        max_length = self._settings.classifier_max_length
        out: list[float] = []
        for start in range(0, len(texts), self._settings.batch_size):
            chunk = list(texts[start : start + self._settings.batch_size])
            encoded = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits.float(), dim=-1)
            out.extend(probs[:, self._ai_index].tolist())
        return out

    # ----------------------------------------------------------------- private
    @staticmethod
    def _resolve_label_index(config, kind: str) -> int:
        """Find the AI (or human) label by name so checkpoints can be swapped."""
        id2label = getattr(config, "id2label", {}) or {}
        labels = {str(k): str(v).strip().lower() for k, v in dict(id2label).items()}
        wanted = _AI_LABELS if kind == "ai" else _HUMAN_LABELS
        for index, label in labels.items():
            if label in wanted:
                return int(index)
        # Fall back to the majority "human" class ordering used by most checkpoints.
        logger.warning("model etiketleri tanınmadı: %s", labels)
        return 1 if kind == "ai" else 0


class DetectorService:
    """Sentence -> ``p_ai`` with optional neighbour context."""

    def __init__(
        self,
        settings: AIModelSettings,
        backend: ClassifierBackend | None = None,
        *,
        model_id: str = "",
    ) -> None:
        self._settings = settings
        self._model_id = model_id or settings.classifier_model
        self._backend = backend

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def available(self) -> bool:
        return not isinstance(self._backend, NullDetector)

    def load(self) -> None:
        if self._backend is not None and not isinstance(self._backend, NullDetector):
            return
        if not self._settings.use_classifier:
            raise SignalUnavailable("sınıflandırıcı devre dışı (use_classifier=false)")
        self._backend = HFDetector(self._model_id, self._settings)

    def score(self, sentences: Sequence[Sentence]) -> dict[int, ClassifierStats]:
        if self._backend is None:
            self.load()
        backend = self._backend
        if backend is None or isinstance(backend, NullDetector):
            return {}

        span = self._settings.classifier_context_sentences
        # Position, not document index: the list may be filtered.
        texts = [
            " ".join(s.text for s in sentences[max(0, position - span) : position + span + 1])
            for position, _ in enumerate(sentences)
        ]
        probabilities = backend.p_ai(texts)
        if not probabilities:
            return {}
        self._check_usefulness(probabilities)
        return {
            sentence.index: ClassifierStats(
                p_ai=float(probabilities[i]),
                model=self._model_id,
                context_sentences=span,
            )
            for i, sentence in enumerate(sentences)
            if i < len(probabilities)
        }

    @staticmethod
    def _check_usefulness(probabilities: Sequence[float]) -> None:
        """Drop the signal when the head says nothing at all.

        An English-only checkpoint scores Turkish text at ~0% in every
        sentence, and a saturated head returns ~99% everywhere. Both are worse
        than no signal: they would pull every sentence toward (or away from) the
        same value while looking like evidence.
        """
        low = min(probabilities)
        high = max(probabilities)
        if high < _USELESS_BELOW:
            raise SignalUnavailable(
                f"sınıflandırıcı hiçbir cümlede %{round(high * 100)} üzerine çıkmadı "
                "(dil uyumsuz olabilir ya da model yalnızca İngilizce eğitilmiş)"
            )
        if low > _USELESS_ABOVE:
            raise SignalUnavailable(
                f"sınıflandırıcı her cümlede %{round(low * 100)} üzerinde "
                "(doygun; anlamlı ayrım üretmiyor)"
            )
