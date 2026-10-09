"""Probability calibration: is the risk score a probability, or just a number?

The gap this closes
-------------------
Every score this tool emits is on a 0-1 scale, which reads like a probability.
It is not one. Nothing in the pipeline maps "risk 0.62" to "62% of the time
this is machine text", and pretending otherwise is the specific failure mode
that turns a ranking into a verdict. Two measured facts make that concrete:

* **Calibration moves without accuracy moving.** Sci-SpanDet reports ECE 0.06 /
  Brier 0.22 with temperature scaling, and shows that *removing* the scaling
  doubles ECE to 0.12 while F1 changes from 80.17 to 80.16 (arXiv:2510.00890,
  *Knowledge-Based Systems* 334:115123). Identical accuracy, twice the
  miscalibration. A threshold that was tuned elsewhere therefore does not
  transfer even when discrimination is perfect.
* **The threshold, not the model, dominates reported F1.** On AITDNA, across
  corpora, F1 swings by **up to 0.8** for a threshold anywhere in the plausible
  policy range tau in [0.1, 0.5], and tau = 0.1-0.2 is called "unacceptably
  high FPR" (arXiv:2606.04906, UKP Lab 2026).
* **The literature does not report agreement on AI involvement at all.** HART,
  AITDNA, LLM-DetectAIve and DETree all omit kappa/alpha for labelling AI
  involvement in academic prose, so there is no published ceiling to calibrate
  against.

What this module does
---------------------
It measures the reliability of an existing local corpus - the sentences
``checker calibrate`` already walks - and reports ECE, Brier and the
reliability bins, plus a fitted temperature. It does not invent a score: with no
corpus there is no calibration, and :func:`reliability_report` says so instead of
returning a confident-looking default.

This is deliberately the same contract as everything else in the tool: the
number comes with its own uncertainty, and its own statement of what it cannot
do. A calibrated probability is still not proof of authorship; ``DISCLAIMER``
says that and keeps saying it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "ReliabilityBin",
    "ReliabilityReport",
    "expected_calibration_error",
    "brier_score",
    "reliability_report",
    "fit_temperature",
    "MIN_SAMPLES",
    "CALIBRATION_EVIDENCE",
]

#: Below this, ECE is not worth reporting. Reliability needs enough samples per
#: bin; below the floor the honest answer is "not measurable", not a number.
MIN_SAMPLES = 30

#: Why calibration is separate from accuracy, with the numbers.
CALIBRATION_EVIDENCE = (
    "ECE/Brier ayrı kalır: sıcaklık ölçeklemeyi kaldırmak F1'i 80.17 -> 80.16 "
    "değiştirirken ECE'yi 0.06 -> 0.12'ye ikiye katlıyor (Sci-SpanDet, "
    "arXiv:2510.00890). Yani ayrım (AUROC/F1) korunurken olasılık bozulur. "
    "AITDNA (arXiv:2606.04906) ise tau in [0.1, 0.5] aralığında F1'in korpusa "
    "göre 0.8 kadar salındığını gösteriyor. Ve HART/AITDNA/LLM-DetectAIve/"
    "DETree hicbiri akademik metinde AI katılımı etiketlemesi icin kappa "
    "raporlamiyor - yani kalibre edilecek yayımlanmış bir tavan yok."
)


def expected_calibration_error(
    pairs: Sequence[tuple[float, float]], bins: int = 10
) -> float | None:
    """Mean |confidence - accuracy| weighted by bin population.

    ``pairs`` are ``(score, was_positive)``. Returns ``None`` when there is not
    enough data, because a number computed from 8 sentences is worse than no
    number at all.
    """
    if len(pairs) < MIN_SAMPLES:
        return None
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for score, label in pairs:
        index = min(bins - 1, max(0, int(min(1.0, max(0.0, score)) * bins)))
        buckets[index].append((score, label))
    total = len(pairs)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_score = sum(s for s, _ in bucket) / len(bucket)
        accuracy = sum(label for _, label in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_score - accuracy)
    return error


def brier_score(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Mean squared error of the score against the binary label."""
    if len(pairs) < MIN_SAMPLES:
        return None
    return sum((score - label) ** 2 for score, label in pairs) / len(pairs)


def fit_temperature(
    pairs: Sequence[tuple[float, float]], *, steps: int = 60
) -> float | None:
    """A one-parameter temperature on the logit scale, fitted by grid search.

    One parameter, not two: a per-bin isotonic fit would happily memorise a
    60-sentence corpus and produce a beautifully calibrated curve that transfers
    nowhere. Temperature can only rescale confidence, which is the honest amount
    of freedom here.
    """
    if len(pairs) < MIN_SAMPLES:
        return None
    best: tuple[float, float] | None = None
    for step in range(1, steps + 1):
        temperature = 0.5 + step * 0.1  # 0.6 .. 6.5
        error = brier_score([(_apply(score, temperature), label) for score, label in pairs])
        if error is None:
            continue
        if best is None or error < best[1]:
            best = (temperature, error)
    return best[0] if best else None


def _apply(score: float, temperature: float) -> float:
    """p = sigmoid(logit(p) / T), guarded at the endpoints."""
    clamped = min(1 - 1e-6, max(1e-6, score))
    logit = _log(clamped / (1 - clamped))
    value = 1 / (1 + _exp(-logit / temperature))
    return min(1.0, max(0.0, value))


def _log(x: float) -> float:  # pragma: no cover - trivial
    import math

    return math.log(x)


def _exp(x: float) -> float:  # pragma: no cover - trivial
    import math

    return math.exp(x)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of the reliability diagram."""

    low: float
    high: float
    count: int
    mean_score: float
    positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "range": [round(self.low, 2), round(self.high, 2)],
            "count": self.count,
            "mean_score": round(self.mean_score, 4),
            "positive_rate": round(self.positive_rate, 4),
            "gap": round(self.mean_score - self.positive_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    """ECE, Brier, the bins, and what they do and do not license."""

    samples: int
    ece: float | None
    brier: float | None
    temperature: float | None
    bins: tuple[ReliabilityBin, ...] = ()
    positive_rate: float = 0.0
    interpretation: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def measured(self) -> bool:
        return self.ece is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "measured": self.measured,
            "ece": round(self.ece, 4) if self.ece is not None else None,
            "brier": round(self.brier, 4) if self.brier is not None else None,
            "temperature": round(self.temperature, 2) if self.temperature is not None else None,
            "positive_rate": round(self.positive_rate, 4),
            "bins": [b.to_dict() for b in self.bins],
            "interpretation": self.interpretation,
            "caveats": list(self.caveats),
            "evidence": CALIBRATION_EVIDENCE,
        }


def reliability_report(pairs: Sequence[tuple[float, float]], *, bins: int = 10) -> ReliabilityReport:
    """Measure how far a 0-1 score is from a probability.

    ``pairs`` are ``(risk_score, was_positive)`` - for a real corpus, "positive"
    means the sentence came from the known-machine corpus. On a thesis you are
    analysing, there is no ground truth, so this can only be run against
    ``checker calibrate`` output, which is exactly why it lives next to it.
    """
    caveats = [
        "Bu ölçüm yalnız etiketli bir korpus üzerinde anlamlıdır; elinizdeki tez için "
        "doğru cevap bilinmediği için burada hesaplanamaz.",
        "Kalibre edilmiş bir olasılık bile suistimal kanıtı değildir. " + CALIBRATION_EVIDENCE,
        "Yayımlanmış Türkçe dedektör AUROC/FPR değerleri **vardır** (AUROC %99.31, "
        "özgüllük %94.16 → FPR %5.84; DOI 10.28948/ngumuh.1930411) ama haber, özet "
        "ve ödev kayıtlarında ölçülmüştür; teze en yakın iki alanda en düşük "
        "sonuçlar oradadır (akademik %95.21, mevzuat-hukuk %94.99). Dahası, aynı "
        "metni %100 insan Türkçe akademik yazıyla deneyen sekiz dedektörün "
        "Justdone'ı %89 AI dediği ölçülmüştür (DOI 10.56493/nkusbmyo.1866431). "
        "Bu yüzden TR işletim noktası başka bir çalışmadan alınmaz; kendi "
        "korporunuzda `checker calibrate` ile ölçülür.",
    ]
    if len(pairs) < MIN_SAMPLES:
        return ReliabilityReport(
            samples=len(pairs),
            ece=None,
            brier=None,
            temperature=None,
            interpretation=(
                f"{len(pairs)} örnek, gereken en az {MIN_SAMPLES}. Kalibrasyon "
                "hesaplanmadı; puanlar olasılık olarak okunamaz."
            ),
            caveats=tuple(caveats),
        )

    rows: list[ReliabilityBin] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (score, label)
            for score, label in pairs
            if low <= score < high or (index == bins - 1 and score == high)
        ]
        if not bucket:
            continue
        rows.append(
            ReliabilityBin(
                low=low,
                high=high,
                count=len(bucket),
                mean_score=sum(s for s, _ in bucket) / len(bucket),
                positive_rate=sum(label for _, label in bucket) / len(bucket),
            )
        )

    ece = expected_calibration_error(pairs, bins)
    brier = brier_score(pairs)
    temperature = fit_temperature(pairs)
    positive_rate = sum(label for _, label in pairs) / len(pairs)

    if ece is not None and ece < 0.05:
        verdict = "iyi kalibre"
    elif ece is not None and ece < 0.10:
        verdict = "kabaca kalibre"
    else:
        verdict = "kalibre değil"
    return ReliabilityReport(
        samples=len(pairs),
        ece=ece,
        brier=brier,
        temperature=temperature,
        bins=tuple(rows),
        positive_rate=positive_rate,
        interpretation=(
            f"ECE {ece:.3f} · Brier {brier:.3f} → skor {verdict}. "
            f"Sıcaklık ölçeklemesi T={temperature:.1f} "
            + (
                "→ skorlar olduğundan keskin; olasılık dilinde okunacaksa düzeltilmeli."
                if temperature and temperature > 1.15
                else "→ skorlar zaten yeterince temkinli."
                if temperature
                else ""
            )
            + f" Korpusta pozitif oran %{positive_rate * 100:.1f}."
        ),
        caveats=tuple(caveats),
    )