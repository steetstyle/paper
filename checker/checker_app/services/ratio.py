"""Likelihood-ratio scoring with two models (the Binoculars family).

Why this exists, with numbers:

* On 135,389 pairs of **edited but human** academic writing, token-statistics
  detectors (entropy, GLTR, log-rank, log-likelihood) flagged **99.8-100%** of
  them as AI, while likelihood-ratio detectors flagged **0.0%** (Binoculars) and
  **0.2%** (LRR) (Park, Jeong & Kim, arXiv:2608.26710).
* On formal legal text the perplexity family collapses: false-positive rates of
  **80.5%** (DetectGPT), **78.3%** (Binoculars) and **61.3%** (Fast-DetectGPT) on
  granted EPO patent claims (arXiv:2607.13044, ICML 2026 AI4Law workshop).

A thesis is exactly the register where those detectors misfire: long, formal,
non-native-in-part, formulaic. So the sentence score that carries the most weight
in this tool is a *disagreement between two models*, not the raw perplexity of
one:

    score = log(PPL_observer) / log(PPL_performer)

Human text is hard for both models but they disagree; machine text is easy for
both and they agree, so the ratio collapses toward 1. It is also why the raw
perplexity keeps only a small weight here and is reported mainly as the
GPTZero-style descriptive statistic it is.

Cost: a second model in memory and one extra forward pass per sentence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from checker_app.config import AIModelSettings
from checker_app.domain.models import RatioStats, Sentence
from checker_app.logging import get_logger
from checker_app.services.perplexity import HFCausalLM, SignalUnavailable

__all__ = ["RatioStats", "LikelihoodRatioService", "observer_models"]

logger = get_logger("checker.ratio")


def observer_models(observer: str, performer: str | None = None) -> tuple[str, str]:
    """Return ``(observer, performer)`` model ids for a language.

    Same-tokenizer pairs are preferred because the ratio is only exactly the
    Binoculars score when both models score identical token sequences.
    """
    if performer:
        return observer, performer
    default = {
        # gpt2 / gpt2-large share the GPT-2 BPE vocabulary exactly.
        "en": (observer, "gpt2-large"),
        # Both Turkish GPT-2 checkpoints use the 50,257-token GPT-2 vocabulary.
        "tr": (observer, "hakanbogan/gpt2-turkish-cased"),
    }
    return default.get("en", (observer, "gpt2-large"))


class LikelihoodRatioService:
    """Score sentences by how much two models agree on their surprisal."""

    def __init__(
        self,
        settings: AIModelSettings,
        *,
        observer_model: str,
        performer_model: str,
        backend_factory=HFCausalLM,
    ) -> None:
        self._settings = settings
        self.observer_model = observer_model
        self.performer_model = performer_model
        self._factory = backend_factory
        self._observer: HFCausalLM | None = None
        self._performer: HFCausalLM | None = None

    # ------------------------------------------------------------------ public
    @property
    def available(self) -> bool:
        return self._observer is not None and self._performer is not None

    def load(self) -> None:
        if self.available:
            return
        if not self._settings.use_perplexity:
            raise SignalUnavailable("likelihood-ratio devre dışı (use_perplexity=false)")
        self._observer = self._factory(self.observer_model, self._settings)
        self._observer.set_batch_size(self._settings.batch_size)
        self._performer = self._factory(self.performer_model, self._settings)
        self._performer.set_batch_size(self._settings.batch_size)
        logger.info(
            "likelihood-ratio modelleri: gözlemci=%s, icra eden=%s",
            self.observer_model,
            self.performer_model,
        )

    def score(self, sentences: Sequence[Sentence]) -> dict[int, RatioStats]:
        if not self.available:
            self.load()
        if self._observer is None or self._performer is None:
            return {}
        context = self._settings.context_sentences
        # Encode once, with the *observer*, and hand the same ids to both models.
        # Scoring each model on its own tokenisation would measure the two
        # tokenizers rather than the authorship: measured AUC 0.375 on Turkish.
        sequences: list[list[int]] = []
        prefix_lengths: list[int] = []
        for position, sentence in enumerate(sentences):
            prefix = (
                " ".join(s.text for s in sentences[max(0, position - context) : position])
                if context
                else ""
            )
            joined = f"{prefix} {sentence.text}" if prefix else sentence.text
            ids, prefix_length = self._observer.encode(
                joined, offset=len(prefix) + (1 if prefix else 0)
            )
            sequences.append(ids)
            prefix_lengths.append(prefix_length)

        observer_scores = self._observer.score_token_ids(sequences, prefix_lengths)
        performer_scores = self._performer.score_token_ids(sequences, prefix_lengths)

        out: dict[int, RatioStats] = {}
        for sentence, left, right in zip(
            sentences, observer_scores, performer_scores, strict=True
        ):
            log_ppl = math.log(left.perplexity) if 0 < left.perplexity < math.inf else 0.0
            log_xppl = math.log(right.perplexity) if 0 < right.perplexity < math.inf else 0.0
            score = log_ppl / log_xppl if abs(log_xppl) > 1e-6 else 1.0
            out[sentence.index] = RatioStats(
                score=round(score, 4),
                observer_perplexity=round(left.perplexity, 3),
                performer_perplexity=round(right.perplexity, 3),
                token_count=min(left.token_count, right.token_count),
                observer_model=self.observer_model,
                performer_model=self.performer_model,
                usable=left.token_count >= 5 and right.token_count >= 5,
            )
        return out
