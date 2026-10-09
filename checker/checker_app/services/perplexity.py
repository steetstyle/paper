"""Perplexity and burstiness with a causal language model.

This is the GPTZero-style core, made sentence-addressable:

* every sentence is scored **conditioned on its context** (the previous
  sentences), because the perplexity of an isolated sentence mostly measures its
  topic, while the perplexity *given the preceding text* measures how
  predictable the sentence is once you are already there - which is what
  distinguishes "expected" from "generated"
* burstiness is the standard deviation of the per-sentence perplexities (and of
  the sentence lengths): human writing is uneven, generated writing is smooth

The model is behind :class:`LMBackend`, so the scoring windowing is unit tested
without downloading 500 MB of weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from checker_app.config import AIModelSettings, CheckerSettings
from checker_app.domain.models import PerplexityStats, Sentence
from checker_app.logging import get_logger

__all__ = [
    "LMRequest",
    "LMScore",
    "LMBackend",
    "NullBackend",
    "HFCausalLM",
    "PerplexityService",
    "SignalUnavailable",
    "burstiness",
    "cv",
]

logger = get_logger("checker.perplexity")


class SignalUnavailable(RuntimeError):
    """Raised when a signal cannot run; the runner degrades instead of failing."""


@dataclass(frozen=True, slots=True)
class LMRequest:
    """Text to score, plus the character offset where scoring must start."""

    text: str
    score_from_char: int = 0


@dataclass(frozen=True, slots=True)
class LMScore:
    """Mean negative log-likelihood (nats/token) over the scored span."""

    nll: float
    token_count: int
    truncated: bool = False
    prefix_tokens: int = 0
    """Tokens of context the scored span was conditioned on."""

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll) if self.nll < 20 else math.inf


class LMBackend(Protocol):
    """Minimal causal-LM surface the service needs."""

    name: str
    context_size: int

    def score(self, requests: Sequence[LMRequest]) -> list[LMScore]:
        """Score a batch of requests."""
        ...


@dataclass(slots=True)
class NullBackend:
    """Used when the model cannot be loaded, so the report degrades cleanly."""

    name: str = "unavailable"
    context_size: int = 0

    def score(self, requests: Sequence[LMRequest]) -> list[LMScore]:
        return [LMScore(nll=0.0, token_count=0) for _ in requests]


class HFCausalLM:
    """``transformers`` causal LM backend, loaded lazily on first use."""

    def __init__(self, model_id: str, settings: AIModelSettings) -> None:
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise SignalUnavailable(
                "torch/transformers kurulu değil. `pip install 'checker-app[ai]'`."
            ) from exc

        from checker_app.services.model_fetch import ensure_model  # noqa: PLC0415

        resolved = ensure_model(model_id, settings.cache_dir) or model_id
        self._torch = torch
        self.model_id = model_id
        self.name = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(resolved)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict[str, object] = {}
        device: Any = self._resolve_device(settings.device)
        if device != "cpu":
            kwargs[self._dtype_arg()] = self._resolve_dtype(settings.dtype, device)
        self.device = device
        self._model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
        self._model.to(device)
        self._model.eval()

        config = self._model.config
        self.context_size = int(
            min(
                getattr(config, "n_positions", 1024) or 1024,
                getattr(config, "max_position_embeddings", 1024) or 1024,
            )
        )
        self.vocab_size = int(getattr(config, "vocab_size", 0) or 0)
        self._pad_id = self._safe_id(
            self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        )
        if self.vocab_size and len(self._tokenizer) > self.vocab_size:
            # Observed in the wild: hakanbogan/gpt2-turkish-cased ships a
            # tokenizer with 50258 entries for a 50257-wide embedding matrix, so
            # its own eos id (50257) is out of range - and a padded batch then
            # makes ``logprobs.gather`` index past the end of the logits, which
            # is a CUDA device assert rather than a wrong number.
            logger.warning(
                "%s: tokenizer %d > model vocab %d; pad ve hedef tokenlar sinir "
                "icine kirpiliyor",
                model_id, len(self._tokenizer), self.vocab_size,
            )

    # ------------------------------------------------------------------ public
    def score(self, requests: Sequence[LMRequest]) -> list[LMScore]:
        if not requests:
            return []
        torch = self._torch
        encoded = self._encode(requests)
        order = sorted(range(len(requests)), key=lambda i: len(encoded[i]["input_ids"]))

        results: list[LMScore | None] = [None] * len(requests)
        batch_size = max(1, self._batch_size)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            batch = [encoded[i] for i in chunk]
            width = max(len(b["input_ids"]) for b in batch)
            input_ids = torch.full((len(batch), width), self._pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, item in enumerate(batch):
                ids = item["input_ids"]
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[row, : len(ids)] = 1
            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)

            with torch.no_grad():
                model = cast(Any, self._model)
                logits = model(input_ids=input_ids, attention_mask=attention).logits
                logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
                targets = input_ids[:, 1:]
                # ``gather`` reads memory: an id outside the logit range is a
                # hard crash (CUDA device assert), not a wrong number.
                valid_ids = targets.clamp_(0, max(0, logprobs.shape[-1] - 1))
                token_nll = -logprobs.gather(2, valid_ids.unsqueeze(-1)).squeeze(-1)

            for row, index in enumerate(chunk):
                item = batch[row]
                mask = self._score_mask(row, item, token_nll, attention)
                count = int(mask.sum().item())
                total = float(token_nll[row][mask].sum().item()) if count else 0.0
                results[index] = LMScore(
                    nll=total / count if count else 0.0,
                    token_count=count,
                    truncated=bool(item["truncated"]),
                    prefix_tokens=int(item["prefix_len"]),
                )
        return [r or LMScore(0.0, 0) for r in results]

    def score_token_ids(
        self,
        sequences: Sequence[Sequence[int]],
        prefix_lengths: Sequence[int] = (),
    ) -> list[LMScore]:
        """Score token sequences supplied by the caller.

        This is what makes a cross-model ratio meaningful: both models must see
        the *same* ids. Two checkpoints with different tokenizers score their own
        tokenisations, and the resulting "ratio" measures the tokenizers, not the
        authorship - measured AUC 0.375 on Turkish before this existed.
        """
        if not sequences:
            return []
        torch = self._torch
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((len(sequences), width), self._pad_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, ids in enumerate(sequences):
            clipped = [self._safe_id(i) for i in ids][-self.context_size :]
            if not clipped:
                clipped = [self._pad_id]
            input_ids[row, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            attention[row, : len(clipped)] = 1
        input_ids = input_ids.to(self.device)
        attention = attention.to(self.device)

        with torch.no_grad():
            model = cast(Any, self._model)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = input_ids[:, 1:].clamp_(0, max(0, logprobs.shape[-1] - 1))
            token_nll = -logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        out: list[LMScore] = []
        for row in range(len(sequences)):
            prefix = int(prefix_lengths[row]) if row < len(prefix_lengths) else 0
            positions = torch.arange(1, token_nll.shape[1] + 1, device=token_nll.device)
            mask = (positions >= prefix + 1) & attention[row, 1:].bool()
            count = int(mask.sum().item())
            total = float(token_nll[row][mask].sum().item()) if count else 0.0
            out.append(
                LMScore(
                    nll=total / count if count else 0.0,
                    token_count=count,
                    prefix_tokens=prefix,
                )
            )
        return out

    def encode(self, text: str, *, offset: int = 0) -> tuple[list[int], int]:
        """Tokenise ``text`` and return ``(ids, prefix_length)`` for a context."""
        prefix_ids = (
            self._tokenizer(text[:offset], add_special_tokens=False)["input_ids"]
            if offset
            else []
        )
        body_ids = self._tokenizer(text[offset:], add_special_tokens=False)["input_ids"]
        budget = max(8, self.context_size - 2)
        if len(prefix_ids) > budget - 8:
            prefix_ids = prefix_ids[-(budget - 8) :]
        room = max(4, budget - len(prefix_ids))
        ids = [self._safe_id(i) for i in (*prefix_ids, *body_ids[:room])]
        return ids or [self._pad_id], len(prefix_ids)

    # ----------------------------------------------------------------- private
    def _resolve_device(self, requested: str) -> str:
        torch = self._torch
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, requested: str, device: str) -> object:
        torch = self._torch
        if requested != "auto":
            return getattr(torch, requested)
        return torch.float16 if device.startswith("cuda") else torch.float32

    def _safe_id(self, token_id: int | None) -> int:
        """Clamp a token id into the model's embedding range."""
        value = int(token_id or 0)
        if self.vocab_size and not 0 <= value < self.vocab_size:
            return 0
        return value

    @staticmethod
    def _dtype_arg() -> str:
        """transformers >= 5 renamed ``torch_dtype`` to ``dtype``."""
        import transformers  # noqa: PLC0415

        major = int(str(transformers.__version__).split(".", 1)[0])
        return "dtype" if major >= 5 else "torch_dtype"

    @property
    def _batch_size(self) -> int:
        # Set through settings at construction time by the service.
        return getattr(self, "_batch_size_value", 8)

    def set_batch_size(self, value: int) -> None:
        self._batch_size_value = max(1, value)

    def _encode(self, requests: Sequence[LMRequest]) -> list[dict[str, Any]]:
        """Tokenise, honouring the model context window.

        The scored span is truncated when it does not fit after its context;
        ``truncated`` is recorded so the report can say the number is partial.
        """
        budget = max(8, self.context_size - 2)
        encoded: list[dict[str, Any]] = []
        for request in requests:
            text = request.text
            boundary = min(request.score_from_char, len(text))
            prefix = text[:boundary]
            body = text[boundary:]

            prefix_ids = self._tokenizer(prefix, add_special_tokens=False)["input_ids"]
            body_ids = self._tokenizer(body, add_special_tokens=False)["input_ids"]
            truncated = False
            room = budget - min(len(prefix_ids), budget - 8)
            if len(prefix_ids) > budget - 8:
                # Keep the most recent context only.
                prefix_ids = prefix_ids[-(budget - 8) :]
                truncated = True
            if len(body_ids) > room:
                body_ids = body_ids[:room]
                truncated = True
            ids = [self._safe_id(i) for i in (*prefix_ids, *body_ids)]
            encoded.append(
                {
                    "input_ids": ids or [self._pad_id],
                    "prefix_len": len(prefix_ids),
                    "truncated": truncated,
                }
            )
        return encoded

    def _score_mask(self, row: int, item: dict[str, Any], token_nll, attention_mask):
        """Positions in the shifted sequence whose *target* is in the body."""
        torch = self._torch
        prefix_len = int(item["prefix_len"])
        width = token_nll.shape[1]
        positions = torch.arange(1, width + 1, device=token_nll.device)
        mask = positions >= prefix_len + 1
        valid = attention_mask[row, 1:].bool()
        return mask & valid


class PerplexityService:
    """Turn sentences into :class:`PerplexityStats`."""

    def __init__(
        self,
        settings: AIModelSettings,
        backend: LMBackend | None = None,
        *,
        model_id: str = "",
        batch_size: int = 8,
    ) -> None:
        self._settings = settings
        self._model_id = model_id
        self._batch_size = batch_size
        self._backend = backend

    # ------------------------------------------------------------------ public
    @property
    def model_id(self) -> str:
        return self._model_id or getattr(self._backend, "name", "unavailable")

    @property
    def available(self) -> bool:
        return not isinstance(self._backend, NullBackend)

    def load(self) -> None:
        """Load the model if it is not loaded yet; raise on failure."""
        if self._backend is not None and not isinstance(self._backend, NullBackend):
            return
        if not self._settings.use_perplexity:
            raise SignalUnavailable("perplexity devre dışı (use_perplexity=false)")
        backend = HFCausalLM(self._model_id, self._settings)
        backend.set_batch_size(self._batch_size)
        self._backend = backend

    def score(self, sentences: Sequence[Sentence]) -> dict[int, PerplexityStats]:
        """Perplexity per sentence index, conditioned on preceding sentences."""
        if self._backend is None:
            self.load()
        backend = self._backend
        if backend is None or isinstance(backend, NullBackend):
            return {}

        context = self._settings.context_sentences
        requests: list[LMRequest] = []
        for position, sentence in enumerate(sentences):
            # ``position``, never ``sentence.index``: callers hand over filtered
            # lists (headings removed), and slicing by document index would make
            # a sentence its own context - which reports perplexity ~1 for all.
            prefix = (
                " ".join(s.text for s in sentences[max(0, position - context) : position])
                if context
                else ""
            )
            joined = f"{prefix} {sentence.text}" if prefix else sentence.text
            requests.append(
                LMRequest(text=joined, score_from_char=len(prefix) + (1 if prefix else 0))
            )

        scores = backend.score(requests)
        model_name = self.model_id
        out: dict[int, PerplexityStats] = {}
        for sentence, score in zip(sentences, scores, strict=True):
            ppl = score.perplexity
            out[sentence.index] = PerplexityStats(
                perplexity=ppl,
                mean_nll=score.nll,
                token_count=score.token_count,
                context_tokens=score.prefix_tokens,
                log10_perplexity=(math.log10(ppl) if 0 < ppl < math.inf else 0.0),
                model=model_name,
                truncated=score.truncated,
            )
        return out

    def backend(self) -> LMBackend:
        if self._backend is None:
            self.load()
        return self._backend or NullBackend()


def burstiness(values: Sequence[float]) -> float:
    """Standard deviation of a sequence (0 for fewer than two values)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def cv(values: Sequence[float]) -> float:
    """Coefficient of variation: burstiness relative to the mean."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return burstiness(values) / abs(mean)


def build_perplexity_service(
    settings: CheckerSettings,
    language: str,
    backend: LMBackend | None = None,
) -> PerplexityService:
    return PerplexityService(
        settings.ai,
        backend=backend,
        model_id=settings.language_model(language),
        batch_size=settings.ai.batch_size,
    )
