"""Likelihood-ratio sinyalinin gerçekten ayrım yaptığını ölçer.

Araştırma, bu ailenin (Binoculars/LRR) düzenlenmiş ancak insan tarafından
yazılmış akademik metinlerde neredeyse sıfır yanlış pozitif verdiğini
söylüyor (arXiv:2608.26710: Binoculars %0.0, LRR %0.2). Türkçe için ise
yayımlanmış bir AUROC/FPR **yok**; bu yüzden ölçümü kendimiz yapıyoruz.

Çıktı: insan ve AI-benzeri metinler için oran dağılımları ve ayrım ölçütleri
(medyan farkı, AUC, verilen bir eşikte FPR).

    ../paper/.venv/bin/python scripts/measure_ratio.py
    .venv/bin/python scripts/measure_ratio.py --lang en
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checker_app.config import CheckerSettings  # noqa: E402
from checker_app.domain.segmentation import SentenceSplitter  # noqa: E402
from checker_app.services.ratio import LikelihoodRatioService, observer_models  # noqa: E402

TR_AI = [
    "Bu çalışmada, öğrencilerin akademik yazım performansının yapay zeka araçlarıyla desteklenmesi üzerine kapsamlı bir inceleme sunulmaktadır. Ayrıca, bu tür araçların kullanımının öğrenme süreçleri üzerindeki etkisi detaylı bir analiz ile ele alınmıştır.",
    "Sürdürülebilir kent planlaması, artan nüfus baskısı karşısında giderek önemli bir araştırma alanı haline gelmektedir. Bu makalede, yeşil altyapı stratejilerinin şehirlerin iklim direncine etkisi değerlendirilmiştir.",
    "Dijital sağlık uygulamalarının hasta memnuniyetine etkisi konusunda artan bir ilgi bulunmaktadır. Çalışmada, mobil uygulamaların takip süreçlerini kolaylaştırdığı ve hasta beklentilerini olumlu yönde etkilediği bulunmuştur.",
]
TR_HUMAN = [
    "Dün akşam laboratuvarda oturup veriye baktım. Sonuçlar beklediğim gibi değil çıktı, biraz can sıkıcı ama en azından nedenini bulduk gibi. Bence asıl sorun veri temizliğinde.",
    "Bu tezde örneklem yönteminin neden bu şekilde seçildiğini açıklamak istiyorum. Katılımcı sayısının az olması bir sınırlılık ama yine de elde ettiğimiz bulgular anlamlı görünmektedir.",
    "Türkçe literatürde konuyla ilgili sınırlı sayıda çalışma bulunmaktadır. Yılmaz ve arkadaşlarının 2019 tarihli makalesi, konuyu kuramsal açıdan ele almaktadır. Bu makalenin eksik bıraktığı nokta ise örneklem büyüklüğüdür.",
]
EN_AI = [
    "Large language models have significantly transformed the landscape of academic writing. This paper provides a comprehensive overview of the role these tools play in contemporary educational settings.",
    "Sustainable urban planning has become an increasingly important research area in response to growing population pressures. The present study examines the impact of green infrastructure strategies on climate resilience in cities.",
    "The impact of digital health applications on patient satisfaction has attracted considerable attention in recent years. The research demonstrates that mobile applications streamline tracking processes.",
]
EN_HUMAN = [
    "I spent two hours in the lab last night going through the recordings again. Most of them are fine, but about one in ten has background noise that would throw the classifier off. I am not going to fix all of them by hand.",
    "This thesis argues that the sample was too small to support the original claim. Attrition is the main threat here, and I do not think a statistical fix is available. Instead, the analysis was repeated on the complete-case subset.",
    "Turkish work on this topic is surprisingly thin. Yilmaz et al. (2019) treat it theoretically, but their sample was small and their instrument has not been validated. Karaca's field study is more useful.",
]
SAMPLES = {"tr": (TR_AI, TR_HUMAN), "en": (EN_AI, EN_HUMAN)}


def auc(human: list[float], machine: list[float]) -> float:
    """P(machine score < human score) - 0.5 means no separation."""
    wins = sum(1 for h in human for m in machine if m < h)
    ties = sum(1 for h in human for m in machine if m == h)
    return (wins + 0.5 * ties) / (len(human) * len(machine))


def measure(lang: str, center: float, scale: float) -> None:
    settings = CheckerSettings()
    observer, performer = observer_models(
        settings.language_model(lang), settings.ratio_model(lang)
    )
    print(f"\n=== dil={lang} gözlemci={observer} icra={performer}")
    print(f"    işletim noktası: center={center} scale={scale}")
    splitter = SentenceSplitter(settings.segmentation)
    service = LikelihoodRatioService(
        settings.ai, observer_model=observer, performer_model=performer
    )
    service.load()

    results: dict[str, list[tuple[float, float]]] = {}
    for label, texts in zip(("ai", "insan"), SAMPLES[lang], strict=True):
        rows: list[tuple[float, float]] = []
        for text in texts:
            segmentation = splitter.split(text)
            sentences = [s for s in segmentation.sentences if s.word_count >= 12]
            if not sentences:
                continue
            for stats in service.score(sentences).values():
                if stats.usable:
                    rows.append((stats.score, stats.machine_likelihood(center, scale)))
        results[label] = rows
        if not rows:
            print(f"  {label}: ölçülebilir cümle yok")
            continue
        ratios = [r for r, _ in rows]
        likes = [m for _, m in rows]
        print(
            f"  {label:5s} n={len(rows):2d} oran medyan={statistics.median(ratios):.3f} "
            f"min={min(ratios):.3f} max={max(ratios):.3f} · "
            f"makine-benzeri medyan={statistics.median(likes):.3f}"
        )

    human = [r for r, _ in results.get("insan", [])]
    machine = [r for r, _ in results.get("ai", [])]
    if not human or not machine:
        print("  ayrım ölçülemedi")
        return
    print(f"  AUC(oran) = {auc(human, machine):.3f}  (0.5 = ayrım yok, 1.0 = mükemmel)")
    flagged = [h for h in human if h < center]
    print(
        f"  eşik {center}: insan metninde işaretlenen oran = {len(flagged)}/{len(human)} "
        f"(FPR = {len(flagged) / len(human):.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="tr", choices=["tr", "en", "both"])
    parser.add_argument("--center", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    args = parser.parse_args()

    settings = CheckerSettings()
    langs = ["tr", "en"] if args.lang == "both" else [args.lang]
    centers = {lang: args.center for lang in langs} if args.center is not None else {
        lang: settings.ratio_threshold(lang) for lang in langs
    }
    scale = args.scale if args.scale is not None else settings.scoring.ratio_scale
    for lang in langs:
        measure(lang, centers[lang], scale)
    print(
        "\nNot: Türkçe için yayımlanmış bir AUROC/FPR yok (2025-26 literatüründe "
        "Türkçe dedektör ölçümü bulunmadı). Buradaki sayılar bu araç için, bu "
        "küçük örneklem üzerinde ölçülmüştür."
    )


if __name__ == "__main__":
    main()