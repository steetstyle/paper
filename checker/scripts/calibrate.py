"""Kalibrasyon ölçümü: perplexity dağılımlarını gerçek örneklerden türetir.

Bu bir araç değil, ayar dosyasıdır: `checker-app` içindeki
`ppl_center_log10` / `ppl_scale_log10` varsayılanlarını buradaki dağılımlara
göre seçmek için kullanılır. Çalıştırmak için:

    ../paper/.venv/bin/python scripts/calibrate.py

Sonuçlar README'deki "Kalibrasyon" bölümüne elle kopyalanır. Bu küçük bir
örneklemdir (dil başına 6 belge) ve mutlak bir doğruluk iddiası değildir.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checker_app.config import CheckerSettings  # noqa: E402
from checker_app.domain.segmentation import SentenceSplitter  # noqa: E402
from checker_app.services.perplexity import PerplexityService  # noqa: E402

TR_AI = [
    """Yapay zeka destekli öğrenme sistemlerinin eğitimdeki rolü giderek önem kazanmaktadır. Bu çalışmada, söz konusu sistemlerin öğrenci başarısı üzerindeki etkisi kapsamlı bir şekilde incelenmiştir. Ayrıca, öğretmen algısının bu süreçteki yeri ayrıntılı olarak ele alınmıştır. Sonuç olarak, teknoloji entegrasyonunun pedagojik kararlarla birlikte planlanması gerektiği vurgulanmaktadır.""",
    """Sürdürülebilir kent planlaması, artan nüfus baskısı karşısında giderek önemli bir araştırma alanı haline gelmektedir. Bu makalede, yeşil altyapı stratejilerinin şehirlerin iklim direncine etkisi değerlendirilmiştir. Öte yandan, bu stratejilerin uygulanabilirliği konusunda önemli engellerin bulunduğu da vurgulanmaktadır. Dolayısıyla, planlama süreçlerinde uzun vadeli düşünmenin gerekliliği ortaya konulmaktadır.""",
    """Dijital sağlık uygulamalarının hasta memnuniyetine etkisi konusunda artan bir ilgi bulunmaktadır. Çalışmada, mobil uygulamaların takip süreçlerini kolaylaştırdığı ve hasta beklentilerini olumlu yönde etkilediği bulunmuştur. Bunun yanı sıra, veri gizliliği konusundaki kaygıların bu uygulamaların kullanımını sınırlayabileceği de belirtilmiştir. Genel olarak, teknolojik çözümlerin hasta memnuniyetini artırma potansiyeli taşıdığı söylenebilir.""",
]

TR_HUMAN = [
    """Dün akşam laboratuvarda oturup veriye baktım. Sonuçlar beklediğim gibi değil çıktı, biraz can sıkıcı ama en azından nedenini bulduk gibi. Bence asıl sorun veri temizliğinde. Önceki dönemde topladığımız kayıtların bir kısmı bozuk, onları temizlemeden çalıştırmak yanlış olur. Yarın sabah tekrar bakacağım.""",
    """Bu tezde örneklem yönteminin neden bu şekilde seçildiğini açıklamak istiyorum. Katılımcı sayısının az olması bir sınırlılık ama yine de elde ettiğimiz bulgular anlamlı. 2022 bahar döneminde toplanan verilerden yararlanılmıştır. Bununla birlikte, farklı bir üniversiteden de veri alınması planlanmaktadır. Bu çalışmanın ikinci bölümünde ele alınan yöntem, üçüncü bölümdeki analizlere temel oluşturmaktadır.""",
    """Türkçe literatürde konuyla ilgili sınırlı sayıda çalışma bulunmaktadır. Yılmaz ve arkadaşlarının 2019 tarihli makalesi, konuyu kuramsal açıdan ele almaktadır. Bu makalenin eksik bıraktığı nokta, örneklem büyüklüğüdür. Ayrıca Karaca tarafından yapılan saha çalışması, benim verilerimle karşılaştırılabilecek bir temel sunmaktadır. Dolayısıyla, bu tezde söz konusu iki çalışmanın eksikleri giderilmeye çalışılmıştır.""",
]

EN_AI = [
    """Large language models have significantly transformed the landscape of academic writing. This paper provides a comprehensive overview of the role these tools play in contemporary educational settings. Moreover, it is important to note that the reliability of automated detection remains a significant concern in the literature. Overall, this comprehensive analysis offers valuable insights for educators navigating this evolving landscape.""",
    """Sustainable urban planning has become an increasingly important research area in response to growing population pressures. The present study examines the impact of green infrastructure strategies on climate resilience in cities. Additionally, it is worth noting that significant barriers to implementation remain. In conclusion, the findings underscore the need for long-term thinking in planning processes.""",
    """The impact of digital health applications on patient satisfaction has attracted considerable attention in recent years. The research demonstrates that mobile applications streamline tracking processes and positively influence patient expectations. However, concerns regarding data privacy may limit the wider adoption of such applications. Overall, technological solutions hold considerable potential for improving patient satisfaction.""",
]

EN_HUMAN = [
    """I spent two hours in the lab last night going through the recordings again. Most of them are fine, but about one in ten has background noise that would throw the classifier off. I am not going to fix all of them by hand. The plan is to drop anything shorter than thirty seconds and re-run the pipeline in the morning. If the numbers still look bad, I will write to the lab manager about the recording setup instead of quietly patching the data.""",
    """This thesis argues that the sample was too small to support the original claim. The pre-registration document from 2022 lists 42 participants, of whom 31 completed both sessions. Attrition is the main threat here, and I do not think a statistical fix is available. Instead, the analysis was repeated on the complete-case subset. Results are reported in Chapter 4 and discussed with respect to the earlier literature.""",
    """Turkish work on this topic is surprisingly thin. Yilmaz et al. (2019) treat it theoretically, but their sample was small and their instrument has not been validated. Karaca's field study is more useful, since it reports the same measurements we collected. Both studies are used here as baselines. Where their conclusions conflict, the disagreement is discussed rather than resolved by choosing the more convenient source.""",
]


def measure(language: str, model_id: str, samples: list[str], label: str) -> None:
    settings = CheckerSettings()
    splitter = SentenceSplitter(settings.segmentation)
    service = PerplexityService(settings.ai, model_id=model_id)
    values: list[float] = []
    print(f"\n### {label} ({language}, {model_id})")
    for text in samples:
        result = splitter.split(text)
        sentences = [s for s in result.sentences if s.location.block_type.value == "prose"]
        scores = service.score(sentences)
        for index in sorted(scores):
            stats = scores[index]
            if stats.token_count >= 5:
                values.append(stats.perplexity)
            print(
                f"  ppl={stats.perplexity:8.1f} log10={stats.log10_perplexity:.2f} "
                f"{text[:0]}{_first_words(sentences, index)}"
            )
    values.sort()
    print(
        f"  -> n={len(values)} min={values[0]:.1f} p25={statistics.quantiles(values, n=4)[0]:.1f} "
        f"median={statistics.median(values):.1f} "
        f"p75={statistics.quantiles(values, n=4)[2]:.1f} max={values[-1]:.1f} "
        f"mean={statistics.fmean(values):.1f}"
    )


def _first_words(sentences: list, index: int) -> str:
    for sentence in sentences:
        if sentence.index == index:
            return sentence.snippet(60)
    return ""


def main() -> None:
    settings = CheckerSettings()
    measure("tr", settings.ai.perplexity_model_tr, TR_AI, "TR/ai-benzeri")
    measure("tr", settings.ai.perplexity_model_tr, TR_HUMAN, "TR/insan")
    measure("en", settings.ai.perplexity_model_en, EN_AI, "EN/ai-benzeri")
    measure("en", settings.ai.perplexity_model_en, EN_HUMAN, "EN/insan")


if __name__ == "__main__":
    main()
