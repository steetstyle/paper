"""Calibrating the operating point on a *local* corpus.

Why this exists, with numbers:

* On RAID, naive thresholds give false-positive rates of 23-100% depending on
  the detector (GLTR 99.3%, LLMDet 96.0%, Fast-DetectGPT 23.2%,
  RoBERTa-Large 2.9%, Binoculars 0.0%) at tau=0.5 (arXiv:2405.07940). There is no
  threshold that transfers between corpora.
* On 135,389 pairs of *edited but human* academic writing, the likelihood-ratio
  detectors flagged 0.0-0.2% while entropy/GLTR/log-rank flagged 99.8-100%
  (arXiv:2608.26710). The difference is calibration to the register at hand.
* There is **no published AUROC or FPR for any detector on Turkish**; the whole
  Turkish detection literature is one 2024 preprint plus one thesis.

So the only defensible operating point is one measured on your own human text.
This command does exactly that, and reports the direction it measured: on some
model pairs the ratio's sign is *inverted* (measured AUC 0.25 for
gpt2/distilgpt2), and shipping a fixed sign would be a coin flip.

    checker calibrate --human-dir ./kendi-yazilarim --out .checker_cache/point.json
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from checker_app.config import CheckerSettings
from checker_app.domain.enums import BlockType
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.logging import get_logger
from checker_app.services.ratio import LikelihoodRatioService, observer_models
from checker_app.services.reliability import ReliabilityReport, reliability_report

__all__ = [
    "CalibrationResult",
    "CalibrationPoint",
    "calibrate",
    "load_calibration",
    "save_calibration",
]

logger = get_logger("checker.calibrate")


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """A measured operating point for the likelihood-ratio signal."""

    language: str
    center: float
    scale: float
    observer: str
    performer: str
    sentences: int
    median_ratio: float
    q10: float
    q90: float
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "center": round(self.center, 4),
            "scale": self.scale,
            "observer": self.observer,
            "performer": self.performer,
            "sentences": self.sentences,
            "median_ratio": round(self.median_ratio, 4),
            "q10": round(self.q10, 4),
            "q90": round(self.q90, 4),
            "note": self.note,
        }


@dataclass(slots=True)
class CalibrationResult:
    """What the corpus said about the signal."""

    human: CalibrationPoint | None = None
    machine: CalibrationPoint | None = None
    auc: float | None = None
    direction: str = "unknown"
    fpr_at_center: float | None = None
    reliability: ReliabilityReport | None = None
    """Is the signal's score a probability? Measured on the same corpus.

    Two published numbers make this worth separating from the AUC: removing
    temperature scaling leaves F1 at 80.17 -> 80.16 while doubling ECE from
    0.06 to 0.12 (arXiv:2510.00890), and the threshold alone swings F1 by up to
    0.8 across corpora for tau in [0.1, 0.5] (arXiv:2606.04906). A perfect AUC
    does not license reading the score as a probability.
    """

    warnings: list[str] = field(default_factory=list)
    files: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "human": self.human.to_dict() if self.human else None,
            "machine_like_reference": self.machine.to_dict() if self.machine else None,
            "auc": round(self.auc, 4) if self.auc is not None else None,
            "direction": self.direction,
            "fpr_at_center": round(self.fpr_at_center, 4) if self.fpr_at_center else None,
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "files": self.files,
            "warnings": self.warnings,
            "recommended_center": (
                round(
                    (self.human.center + self.machine.center) / 2, 4
                )
                if self.human and self.machine
                else (self.human.center if self.human else None)
            ),
        }


def _sentences(text: str, splitter: SentenceSplitter, min_words: int) -> list:
    result = splitter.split(text)
    return [
        s
        for s in result.sentences
        if s.location.block_type is BlockType.PROSE and s.word_count >= min_words
    ]


def _measure(
    texts: Sequence[str],
    language: str,
    settings: CheckerSettings,
    service: LikelihoodRatioService | None = None,
) -> tuple[CalibrationPoint | None, list[float]]:
    splitter = SentenceSplitter(settings.segmentation)
    rows: list[float] = []
    observer, performer = observer_models(
        settings.language_model(language), settings.ratio_model(language)
    )
    own = service is None
    if service is None:
        service = LikelihoodRatioService(
            settings.ai, observer_model=observer, performer_model=performer
        )
        service.load()
    assert service is not None

    for text in texts:
        sentences = _sentences(text, splitter, settings.ai.min_ai_words)
        if not sentences:
            continue
        for stats in service.score(sentences).values():
            if stats.usable:
                rows.append(stats.score)

    if not rows:
        return None, []
    rows.sort()
    point = CalibrationPoint(
        language=language,
        center=statistics.median(rows),
        scale=settings.scoring.ratio_scale,
        observer=observer,
        performer=performer,
        sentences=len(rows),
        median_ratio=statistics.median(rows),
        q10=_quantile(rows, 0.10),
        q90=_quantile(rows, 0.90),
        note="insan metni" if not own else "",
    )
    return point, rows


def _auc(human: Sequence[float], machine: Sequence[float]) -> float:
    wins = sum(1 for h in human for m in machine if m < h)
    ties = sum(1 for h in human for m in machine if m == h)
    return (wins + 0.5 * ties) / (len(human) * len(machine))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def calibrate(
    settings: CheckerSettings,
    *,
    human_files: Sequence[Path],
    machine_files: Sequence[Path] = (),
    language: str | None = None,
) -> CalibrationResult:
    """Measure the signal on local human text (and optionally AI text)."""
    result = CalibrationResult(files=len(human_files))
    if not human_files:
        result.warnings.append("insan dosyası verilmedi; ölçüm yapılamaz")
        return result

    texts = []
    for path in human_files:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:  # pragma: no cover - filesystem dependent
            result.warnings.append(f"okunamadı {path}: {exc}")

    sample = " ".join(texts[:5])
    from checker_app.domain.text import language_confidence  # noqa: PLC0415

    lang = language or language_confidence(sample)[0]
    if lang not in {"tr", "en"}:
        lang = settings.language_model("en") and "en" or "en"
        result.warnings.append("dil belirlenemedi, 'en' varsayıldı")

    human, human_rows = _measure(texts, lang, settings)
    machine_rows: list[float] = []
    result.human = human
    if human is None:
        result.warnings.append("ölçülebilir cümle bulunamadı (cümleler çok kısa olabilir)")
        return result

    if machine_files:
        machine_texts = []
        for path in machine_files:
            try:
                machine_texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:  # pragma: no cover - filesystem dependent
                result.warnings.append(f"okunamadı {path}: {exc}")
        machine, machine_rows = _measure(machine_texts, lang, settings)
        result.machine = machine

    center = settings.ratio_threshold(lang)
    result.fpr_at_center = (
        sum(1 for row in human_rows if row < center) / len(human_rows) if human_rows else None
    )
    if machine_rows and human_rows:
        result.auc = _auc(human_rows, machine_rows)
        result.direction = (
            "düşük oran = makine-benzeri"
            if result.auc >= 0.5
            else "TERS: yüksek oran = makine-benzeri"
        )
        if result.auc < 0.5:
            result.warnings.append(
                "Bu model çifti için sinyal yönü ters çıkıyor; sabit yön varsaymak "
                "zarar verir. Merkezi iki dağılımın ortasına alın ve yönü elle "
                "doğrulayın."
            )
    if result.fpr_at_center is not None and result.fpr_at_center > 0.05:
        result.warnings.append(
            f"verilen merkezde ({center}) insan metninde işaretlenme oranı "
            f"%{round(result.fpr_at_center * 100)}; %5'in üzerinde (RAID: naive "
            "eşiklerde FPR %23-100 arasında)."
        )

    if machine_rows:
        # Reliability needs both classes. Without a machine corpus there is no
        # label, and an ECE computed against an assumed label would be fiction.
        inverted = result.auc is not None and result.auc < 0.5
        sign = -1.0 if inverted else 1.0
        scale = max(1e-6, settings.scoring.ratio_scale)

        def as_probability(value: float) -> float:
            return 1 / (1 + pow(2.718281828, (-sign * (value - center)) / scale))

        pairs: list[tuple[float, float]] = [(as_probability(v), 0.0) for v in human_rows]
        pairs.extend((as_probability(v), 1.0) for v in machine_rows)
        result.reliability = reliability_report(pairs)

    return result


def save_calibration(path: Path, result: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))