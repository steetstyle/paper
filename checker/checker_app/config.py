"""Typed settings (12-factor, environment driven).

Every knob the analysis depends on lives here so a scan is reproducible: the
same text plus the same settings always produce the same report, and the report
records the settings that were used.

Each sub-settings object owns an environment prefix so names never collide
(``CHECKER_SEG_*``, ``CHECKER_AI_*``, ``CHECKER_PLAG_*`` …).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILES = (".env", "../.env")

Lang = Literal["auto", "tr", "en"]


def _cfg(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SegmentationSettings(BaseSettings):
    """How the document is cut into located sentences."""

    model_config = _cfg("CHECKER_SEG_")

    min_sentence_chars: int = Field(
        default=15,
        ge=0,
        description="Fragments shorter than this are merged into the previous sentence.",
    )
    split_inside_quotes: bool = Field(
        default=False,
        description="Allow a boundary inside an open quote or parenthesis.",
    )
    treat_formfeed_as_page_break: bool = True
    detect_language: bool = Field(
        default=True,
        description="Auto-detect tr/en from character and stopword frequency.",
    )
    keep_short_fragments_as_headings: bool = True


class StylometrySettings(BaseSettings):
    """Rule-based, model-free signals. Always available."""

    model_config = _cfg("CHECKER_STY_")

    enabled: bool = True
    lexicon_languages: list[str] = Field(default_factory=lambda: ["tr", "en"])
    formal_register_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Informal-contraction rate below this counts as 'too clean'.",
    )
    min_sentence_words: int = Field(
        default=4, ge=1, description="Shorter sentences skip style statistics."
    )


class AIModelSettings(BaseSettings):
    """Perplexity / burstiness (causal LM) and the classifier head."""

    model_config = _cfg("CHECKER_AI_")

    use_perplexity: bool = True
    use_classifier: bool = True
    use_ratio: bool = Field(
        default=True,
        description=(
            "Two-model likelihood ratio (Binoculars family). This is the highest "
            "weight signal in the scorer because it is the only family measured "
            "near 0% false positive on edited non-native academic writing "
            "(Binoculars 0.0%, LRR 0.2%, vs 99.8-100% for entropy/GLTR/log-rank, "
            "arXiv:2608.26710). Costs a second model in RAM."
        ),
    )

    perplexity_model_tr: str = "ytu-ce-cosmos/turkish-gpt2"
    perplexity_model_en: str = "gpt2"
    # No same-tokenizer Turkish pair is published, so the observer's ids are fed
    # to both models (see LikelihoodRatioService). For English this is the
    # published pairing (gpt2 / gpt2-large), whose reported threshold is 0.901.
    ratio_model_tr: str = "hakanbogan/gpt2-turkish-cased"
    ratio_model_en: str = "gpt2-large"
    classifier_model: str = "Hello-SimpleAI/chatgpt-detector-roberta"

    min_ai_words: int = Field(
        default=12,
        ge=3,
        description=(
            "Sentences shorter than this are not AI-scored. Short text is where "
            "detectors fail and where false positives live: AUROC 0.62 at 10 "
            "tokens vs 0.88 at 256 (arXiv:2509.18880), false-positive modal length "
            "34 words (arXiv:2603.23146)."
        ),
    )

    device: str = Field(default="auto", description="auto | cpu | cuda | cuda:0 | mps")
    dtype: str = Field(default="auto", description="auto | float32 | float16 | bfloat16")

    context_sentences: int = Field(
        default=2,
        ge=0,
        le=16,
        description=(
            "How many preceding sentences condition the perplexity of a sentence. "
            "0 scores every sentence in isolation."
        ),
    )
    max_context_tokens: int = Field(default=768, ge=64, le=4096)
    batch_size: int = Field(default=8, ge=1, le=64)
    classifier_context_sentences: int = Field(
        default=1, ge=0, le=8, description="Neighbour sentences fed to the classifier."
    )
    classifier_max_length: int = Field(default=512, ge=64, le=1024)
    cache_dir: str | None = Field(
        default=None, description="HuggingFace cache; defaults to the standard location."
    )

    @field_validator("perplexity_model_tr", "perplexity_model_en", "classifier_model")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model id must not be empty")
        return v.strip()


class PlagiarismSettings(BaseSettings):
    """Exact n-gram shingle overlap plus optional near-match alignment."""

    model_config = _cfg("CHECKER_PLAG_")

    ngram_size: int = Field(default=4, ge=2, le=12)
    min_match_words: int = Field(
        default=4, ge=2, description="Shorter repeated runs are ignored as coincidental."
    )
    near_match: bool = Field(default=True, description="Also align reworded passages with difflib.")
    near_match_ratio: float = Field(
        default=0.85, ge=0.5, le=1.0, description="Minimum block ratio to report as near-match."
    )
    near_match_min_words: int = Field(default=12, ge=4)
    template_match: bool = Field(
        default=True,
        description=(
            "Also compare digit-folded n-grams: same wording with new numbers is "
            "the usual shape of a reused academic template."
        ),
    )
    template_min_words: int = Field(
        default=12,
        ge=4,
        description="Longer threshold than verbatim, because folded digits match more loosely.",
    )
    strip_diacritics: bool = True
    normalize_digits: bool = Field(
        default=False,
        description="Fold digits in the verbatim index too (very loose; prefer template_match).",
    )
    institutional_min_match_words: int = Field(
        default=5,
        ge=1,
        description=(
            "Turnitin's '5 kelimeden az eşleşmeler hariç' filter, which the Turkish "
            "institutional rule sheets list verbatim."
        ),
    )
    exclude_references: bool = Field(
        default=True, description="'Kaynakça hariç' - reference list never counted."
    )
    include_quotes: bool = Field(
        default=True, description="'Alıntılar dahil' - the headline number."
    )
    containment_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Share of a sentence covered by matched runs before it is flagged.",
    )
    check_code: bool = Field(
        default=True, description="AST/LaTeX comparison for code and math blocks."
    )
    max_source_chars: int = Field(default=4_000_000, ge=1000)
    cache_dir: str = Field(default=".checker_cache", description="Relative to the CWD.")


class ScoringSettings(BaseSettings):
    """How the signals are fused into one risk number."""

    model_config = _cfg("CHECKER_SCORE_")

    # Rebalanced from the 2025-26 measurements, see the docstrings above:
    # raw perplexity drops to a minor term because surprisal detectors reach
    # 61-80% false positives on formal text (EPO claims, arXiv:2607.13044) and
    # ~100% on edited non-native academic writing (arXiv:2608.26710).
    weight_ratio: float = Field(default=0.34, ge=0.0, le=1.0)
    weight_perplexity: float = Field(default=0.08, ge=0.0, le=1.0)
    weight_classifier: float = Field(default=0.22, ge=0.0, le=1.0)
    weight_stylometry: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_uniformity: float = Field(default=0.10, ge=0.0, le=1.0)

    ratio_center: dict[str, float] = Field(
        default_factory=lambda: {"tr": 0.33, "en": 0.90},
        description=(
            "Decision point of the observer/performer ratio, per language. The "
            "published Binoculars threshold is 0.901, but that number belongs to "
            "the gpt2/gpt2-large pair; measured on this tool's pairs the Turkish "
            "midpoint is 0.33 (scripts/measure_ratio.py: AUC 0.75, AI median "
            "0.320 vs human 0.345). Run `checker calibrate` on your own corpus "
            "before trusting any of it."
        ),
    )
    ratio_scale: float = Field(
        default=0.25,
        description="Logistic steepness around ratio_center; smaller = sharper.",
    )
    section_prior_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="How strongly the section role (abstract, method, results) shifts the score.",
    )
    fpr_budget: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description=(
            "The false-positive rate the thresholds are meant to sit at. RAID "
            "shows naive thresholds give 23-100% FPR (arXiv:2405.07940), so the "
            "report states the operating point instead of implying universality."
        ),
    )

    # Logistic calibration of log10(perplexity): below ``center`` looks machine
    # written. Defaults come from scripts/calibrate.py (6 documents per
    # language, 12-15 sentences each): TR median 1.55 human vs 1.38 machine,
    # EN median 1.69 human vs 1.39 machine. Overlap is large by nature, which
    # is exactly why perplexity is only one term of four.
    ppl_center_log10: dict[str, float] = Field(default_factory=lambda: {"tr": 1.50, "en": 1.55})
    ppl_scale_log10: dict[str, float] = Field(default_factory=lambda: {"tr": 0.18, "en": 0.18})
    ppl_relative_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Share of the perplexity term that is within-document relative.",
    )

    plagiarism_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Plagiarism can dominate the final score of a sentence.",
    )
    risk_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.35, "medium": 0.55, "high": 0.75, "critical": 0.90}
    )


class CheckerSettings(BaseSettings):
    """Root settings: sub-settings plus document level options."""

    model_config = SettingsConfigDict(
        env_prefix="CHECKER_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    language: Lang = "auto"
    profile: Literal["document", "thesis"] = Field(
        default="document",
        description=(
            "thesis: section roles drive scoring, reference list is excluded from "
            "overlap, institutional similarity filters are applied."
        ),
    )
    max_chars: int = Field(default=400_000, ge=1000)
    default_encoding: str = "utf-8"
    log_level: str = "INFO"
    disclaimer: bool = Field(
        default=True,
        description="Print the 'this is a risk estimate, not proof' notice.",
    )

    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    stylometry: StylometrySettings = Field(default_factory=StylometrySettings)
    ai: AIModelSettings = Field(default_factory=AIModelSettings)
    plagiarism: PlagiarismSettings = Field(default_factory=PlagiarismSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)

    def ratio_model(self, lang: str) -> str:
        """Second model of the likelihood-ratio pair."""
        return self.ai.ratio_model_tr if lang == "tr" else self.ai.ratio_model_en

    def language_model(self, lang: str) -> str:
        """Causal LM id for ``lang``."""
        return self.ai.perplexity_model_tr if lang == "tr" else self.ai.perplexity_model_en

    def ratio_threshold(self, lang: str) -> float:
        """Operating point of the likelihood-ratio signal for ``lang``."""
        values = self.scoring.ratio_center
        return float(values.get(lang, values.get("en", 0.90)))

    def perplexity_center(self, lang: str) -> float:
        values = self.scoring.ppl_center_log10
        return float(values.get(lang, values.get("en", 1.55)))

    def perplexity_scale(self, lang: str) -> float:
        values = self.scoring.ppl_scale_log10
        return float(values.get(lang, values.get("en", 0.18)))


@functools.lru_cache(maxsize=1)
def get_settings() -> CheckerSettings:
    return CheckerSettings()


def reload_settings() -> CheckerSettings:
    get_settings.cache_clear()
    return get_settings()
