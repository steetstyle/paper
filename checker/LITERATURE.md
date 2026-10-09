# Kanıt dosyası: hangi karar hangi sayıya dayanıyor?

Bu dosya, `checker-app` içindeki her önemli tasarım kararını **2025–2026
literatüründe ölçülmüş bir sayıya** bağlar. Kararın dayanağı yoksa ya da
doğrulanamıyorsa `DOĞRULANAMADI` olarak işaretlenir — uydurma sayı yoktur.

Kaynaklar `checker-literature` projesinde (paper MCP korpusu, 12 makale) ve
`scripts/seed_literature.sh` ile oraya alınmıştır.

**Bu dosyanın birinci turu (AI tespiti + intihal)** yapılandırılmış kaynak
taramasıyla 2025–2026 literatüründen 12 makale üzerine kurulmuştu. İkinci tur
tez yazımına özel boşluklara yöneldi: yapı/discourse sinyalleri, patchwriting ve
kendi kendine intihal ölçütleri, kurumsal temyiz protokolleri. O turun bulguları
§1.5–1.11, §2.2b ve §2.2c'ye işlendi; **iki önemli varsayım yanlış çıktı** ve
kaldırıldı: "PPBS" diye bir metrik yok (§2.2b notu) ve "LWAI" kaynağı
bulunamadı (§5).

**Üçüncü tur** Türkçe kaynaklara ve çapraz dile yöneldi ve **kendi iddiamızı
çürüttü**: "Türkçe'de hiçbir dedektörün yayımlanmış AUROC/FPR değeri yok"
yanlıştı (§1.14). Daha önemlisi, §1.13'teki ölçüm aracın tasarımını doğruluyor:
8 dedektör, %100 insan Türkçe akademik metinde %89'a varan AI iddiasında bulundu
ve aynı içerik Fransızca'da %0, Türkçe çevirisinde %73.25 okundu. İkinci negatif
— Türkçe **tez** benzerlik dağılımı — doğrulandı.

**Dördüncü tur** İngilizce akademik ölçümlere yöneldi ve **tabanların 3 kat
farklı olduğunu** gösterdi: Türkçe tezler %28.7 ± 11.58, temiz İngilizce
doktora tezleri **%9 ± 6** (§1.19). Aynı turda İngilizce kurum eşikleri (§1.20),
alan bazlı AI yaygınlığı (§1.21), model gerektirmeyen iki İngilizce sözlük sinyali
(§1.22) ve tür/uzunluk/L2 uyarıları (§1.23) eklendi. Yayımlanmış İngilizce stil
tablosu olmadığı için insan İngilizce akademik metin referansları **kendi
ölçümümüzle** üretildi ve `PROVENANCE` alanında "kendi ölçüm" olarak işaretlendi
(§1.24).

---

## 1. AI yazım tespiti

### 1.1 Eşik sabit olamaz — çalışma alanında ölçülecek

| Bulgu | Sayı | Kaynak |
|---|---|---|
| RAID'de `τ=0.5` ile FPR | GLTR %99.3 · LLMDet %96.0 · Fast-DetectGPT %23.2 · RoBERTa-Large %2.9 · Binoculars %0.0 | Dugan vd., *RAID*, arXiv:2405.07940 |
| Düzenlenmiş ama **insan** akademik yazımında FPR (135.389 çift) | Log-rank %100 · Entropy %99.8 · GLTR %99.9 · RoBERTa %93.1 · RADAR %88.3 · MAGE %16.7 · LRR **%0.2** · Binoculars **%0.0** | Park vd., *Style as a Confound*, arXiv:2608.26710 |
| Resmî/hukuki metinde FPR (EPO patent iddiaları) | DetectGPT %80.5 · Binoculars %78.3 · Fast-DetectGPT %61.3 | Banerjee, *The Perplexity Trap*, arXiv:2607.13044, ICML 2026 AI4Law |
| Ticari dedektörler, güvenlik bildirileri üzerinde | FPR %0.05–68.6, FNR %0.3–99.6 (N=6.295) | Layton vd., IEEE S&P 2026 |
| Turnitin AI tespiti dil desteği | Yalnız **İngilizce, İspanyolca, Japonca** — Türkçe yok | Turnitin FAQ (2026) |
| Türkçe dedektör performansı | **Yayımlanmış AUROC/FPR yok**; tek benchmark 2024 (arXiv:2408.10724), in-domain F1 0.97–1.00, out-of-domain **0.35–0.42** | Üyük vd., NLP4PI 2024 |

**Kararlarımız**

1. Ham perplexity ağırlığı **0.34 → 0.08** düşürüldü. Surprisal yöntemleri resmî
   metinde %61–80, kendi dilinde yazmayan insan metninde %99–100 yanlış pozitif
   veriyor; tez tam olarak o kayıt.
2. **Likelihood-ratio (iki model) sinyali** en yüksek ağırlığı aldı (0.34):
   ölçülen %0.0–0.2 FPR ile bu kayıtta tek sıfıra yakın aile.
3. Rapor, "kanıtın ne kadarı çalıştı" bilgisini `confidence` olarak taşır; eşikler
   evrensel değil, `CHECKER_SCORE_FPR_BUDGET` ile bir **işletim noktası** olarak
   yazılır (varsayılan %5).
4. `--fail-on` yerine varsayılan **uyarı** modu: RAID ekibi cezai kullanıma
   itiraz ediyor, IEEE S&P 2026 "yüksek riskli kararlar için uygun değil" diyor.
5. Sinyal **işe yaramıyorsa düşürülür**: her cümlede <%5 veya >%95 veren
   sınıflandırıcı bilgi taşımıyor, raporda gerekçesiyle listelenir.
   (Türkçe'de `Hello-SimpleAI/chatgpt-detector-roberta` tam olarak bunu yapar:
   ölçümde her cümlede %0–1.)

### 1.2 Kısa metin tespit edilemez

| Bulgu | Sayı | Kaynak |
|---|---|---|
| AUROC vs uzunluk | 10 token'da 0.62 → 256 token'da 0.88 | Basani & Chen, arXiv:2509.18880 (TMLR 02/2026) |
| Yanlış pozitiflerin uzunluğu | FP medyan 221 kelime, **mod 34 kelime**; FN mod 14 kelime | arXiv:2603.23146, PAN CLEF 2025 |
| CEFR düzeyi | Tüm yöntemler A1'de en kötü, yeterlilik arttıkça iyileşir; Binoculars %68.3 doğruluk | arXiv:2509.18880 |

**Karar:** `min_ai_words = 12`. 12 kelimenin altındaki cümleler AI ölçümüne
girmez, raporda `too_short_for_ai` bulgusu ve `unscored_sentences` sayacıyla
görünür.

### 1.3 Paraphrase ve homoglyph saldırıları

| Bulgu | Sayı | Kaynak |
|---|---|---|
| RAID homoglyph altında doğruluk düşüşü | ortalama **%40.6**; Binoculars −41.9 puan, Originality −75.7 puan | arXiv:2405.07940 |
| Uyarlanabilir paraphrase (StealthRL) | ortalama AUROC 0.789 → **0.432**; Binoculars 0.055; MAGE 0.891 | arXiv:2602.08934 |
| Ticari dedektörler, humanizer sonrası | FNR > %96 (Pangram %96.6, GPTZero %96.1) | arXiv:2608.11256, ACM AILS 2026 |
| Dürüst AI düzeltmesi | hafif düzeltilmiş özetlerde işaretlenme: Pangram %80.1, GPTZero %48.5 | arXiv:2608.11256 |

**Kararlarımız**

* `checker_app/services/integrity.py`: homoglyph, sıfır genişlik karakter, yumuşak
  tire, tekrarlı noktalama tespiti. Bulgu varsa AI sinyalleri "güvenilmez" işaretlenir
  ve gerekçesi rapora yazılır (RAID'in %40.6'sı raporun içinde görünür).
* Paraphrase dayanıklılığı iddiası **yapılmıyor**: skiller ile örtüşen n-gram
  eşleşmesi paraphrase'i yakalamaz (bkz. §2.1); bu araç birebir/şablon/yakın
  eşleştirmeyi raporlar, kaçırma yeteneği iddia etmez.

### 1.4 "Üretildi" değil, "AI katkısı var"

| Bulgu | Sayı | Kaynak |
|---|---|---|
| DETree: AI en az katıldığında bile stil izleri baskın | HART üzerinde AUROC 0.988–0.998, TPR@5%FPR %95.3–99.5 | arXiv:2510.17489 |
| HART katmanları | L1 "AI var mı" AUROC 0.780 · L2 "AI içeriği var mı" 0.711 · L3 "tamamen AI" 0.870 | arXiv:2503.00258 |

**Karar:** Çıktı dili "AI üretti / üretmedi" değil, **"AI katkısı olasılığı"**.
Rapor başlığı ve README bu ayrımı açıkça yapar.

### 1.5 Derece (ne kadar) iddiası yasaklandı

Bu, 1.4'ün mantıklı sonucu ama daha keskin: araç **hiçbir yerde** "metnin %X'i
AI tarafından yazıldı" demez. Gerekçe ölçülmüş:

| Bulgu | Sayı | Kaynak |
|---|---|---|
| GLTR FPR'ı saf insan metninde | %6.83 | arXiv:2502.15666 (APT-Eval) |
| …kelimelerin **%1'i** düzenlendiğinde | **%26.85** | aynı |
| …kozmetik GPT-4o cilasıyla | **%40.87** | aynı |
| LLaMA-2-7B ile hafif cilada | %52.31 | aynı |
| Ticari dedektörlerde en az GPT-4o düzeltmesiyle işaretleme | **%10–75** (dedektöre göre) | aynı |
| Dedektörler düzeltme derecesini ölçebiliyor mu? Hayır: RoBERTa "hafif"→"ağır" arasında yalnız 4.3 puan; Fast-DetectGPT ters (kelimelerin %1'i düzenlendiğinde %10.07, %75'inde %9.59) | — | aynı |

**Karar:** `AI_DEGREE_CAVEAT` her raporun içinde taşınır (JSON `degree_caveat`,
MCP `verdict.caveats.degree`, `disclaimer` kaynağı) — README'de değil, çünkü
insanların ekran görüntüsü aldığı yer cümle listesi.

### 1.6 %20 raporlama tabanı — belge hükmü için

Turnitin'in kendi AIW-2 protokolü (Ağu. 2024) bu eşiği kullanır ve nedenini
sayıyla verir:

| Ölçüm | Değer | Örneklem |
|---|---|---|
| Belge FPR (AIW-2) | **%0.51** | 719.877 pre-2019 insan öğrenci metni |
| Cümle FPR (AIW-2) | **%0.33** | aynı |
| Belge geri çağırma (AIW-2) | %91.18 | 2.970 belge (saf insan / saf AI / karışık) |

**Karar:** `AI_REPORTING_FLOOR = 0.20`. Pay altındaysa belge düzeyinde AI puanı
**verilmez**; yalnız konumlu cümle bulguları raporlanır. Bu, Turnitin'in
belge/cümle ayrımını birebir taklit eder: cümleler belgeden önce raporlanır.
`DocumentScores.ai_share` **saf `ai_score`** eşiğiyle hesaplanır — birleşik risk
içinde intihal de vardır ve atıfsız eşleşme dolu bir belgeye AI payı kazandırmaz.

### 1.7 İnsan tabanı: jüri de bu işte zayıf

| Bulgu | Sayı | Kaynak |
|---|---|---|
| 63 öğretim üyesi, Alman tezlerinden 200–300 kelimelik parçalar | AI metninde **%57**, insan metninde **%64** doğru | DOI 10.1016/j.iree.2025.100321 |
| Profesyonel düzeydeki AI metni | doğruluk **%20'nin altında** | aynı |
| İnsan ile makine dedektör arasında anlamlı fark | **yok** | aynı |
| Cümle düzeyi yerelleştirme, Binoculars ailesi | F1@K **0.608** | 2025.findings-ijcnlp.48 |
| Tez alanı, diğer alanlardan zor (LLM-RR F1) | DeBERTa: tez **85.88**, hikâye 90.05, deneme 60.66 | arXiv:2410.14259 |

**Karar:** `AI_HUMAN_BASELINE` her raporla birlikte döner. Okuyucu "AI %0.39"
gördüğünde karşılaştırma yapabilmeli; bu sayı olmadan rakamın anlamı yok.

### 1.8 Kurumsal uygulama: dedektörler kapatılıyor

| Kurum | Karar | Yıl |
|---|---|---|
| Vanderbilt | Turnitin AI dedektörü devre dışı | 2023 |
| UCLA, UC San Diego | aynı | 2024 |
| University of Waterloo | aynı | Eylül 2025 |
| Curtin University | aynı | Ocak 2026 |

İngiltere Ombudsmanı **CS072502** (Temmuz 2025): öğrenci yalnız Turnitin çıktısı
görülmeden, savunmaya çağrıldı; şikâyet **"haklı" bulundu**, çünkü sağlayıcı
Turnitin'in ana dilini başka bir dilde yazanlara karşı daha güvenilmez
çalıştığını değerlendirmemişti.

Adelphi Üniversitesi (2025): tek başına dedektör puanı "daha fazla inceleme
gerekçesi"dir, yeterli kanıt değildir; doğrulayıcı özellikler arasında
**halüsinasyonlu/eksik kaynaklar** sayılır → bu, §2.5'in referans denetimini
meşru bir çıktı yapan kurumsal metin.

**Karar:** `AI_DEGREE_CAVEAT` ve `disclaimer` kaynağı bu kapanışları da sayıyla
birlikte anar.

### 1.9 Söylem (discourse) profili — düzenlemeden sonra ayakta kalan tek sinyal

| Bulgu | Sayı | Kaynak |
|---|---|---|
| İnsan yazarlar **Attribution** ve **Temporal** ilişkilerini fazla kullanır; LLM'ler **Elaboration** ve **Cause** | yön ölçülmüş | arXiv:2604.04932 (RACE, ACL 2026) |
| Yapı, düzenlemeden sonra korunur | insan→AI cilalama kosinüs benzerliği **0.92 ± 0.08**; insan→humanized 0.86 ± 0.12; insan→LLM 0.84 ± 0.12 | aynı |
| Söylem motifleri eklenince HC3 F1 | 0.67 → **0.73** (GAT); Longformer ile 0.97 → **0.98** | arXiv:2402.10586 |
| Motifler **tek başına** zayıf | F1 0.55 / 0.58 / 0.49 / 0.61 (dört test kümesi) | aynı |
| Uzun belgede söylem sinyali bozulmaz | 1/3/5 üretilmiş paragrafta LF 0.30 → 0.45; LF+Motifs 0.43 → **0.71** | aynı |

**Karar:** `services/discourse.py` — dört ilişki sınıfı TR/EN bağlaçlarıyla
sayılır, 100 kelimeye normalize edilir, yön okuması verilir. Ağırlığı küçük
kalır ve **hüküm üretmez**, çünkü motiflerin tek başına ölçülen F1'si 0.49–0.61.

### 1.10 Leksikal zenginlik: en güçlü özellik alanı, ama yönü kararsız

284 özelliklikli ablation (27 LLM, 10 alan; arXiv:2606.04177):

| Düşürülen özellik alanı | Alan içi Δ | Alan dışı Δ |
|---|---|---|
| **Leksikal zenginlik (3 özellik)** | **−13.1 F1** | **−27.7 F1** |
| Bilgi-kuramsal (2) | −1.8 | — |
| POS (58) | −0.4 | **+0.7** (zararlı) |
| Yüzey (6) / okunabilirlik (2) / varlık (19) / morfoloji (2) / bağımlılık (47) / anlam (16) / duygu (36) / psikolinguistik (77) | ±0.23 | — |

Alan dışında **yalnız leksikal zenginlik** 284 özelliğin tamamını geçiyor
(+14.29 F1). Ancak yön yardım türüne göre değişiyor (arXiv:2410.14259,
64.304 belge):

| Rol | Sözlü zenginlik (0–1) |
|---|---|
| İnsan yazar | 0.59 ± 0.07 |
| **LLM cilacı (polisher)** | **0.61 ± 0.07** |
| LLM üretici (creator) | 0.52 ± 0.06 |
| LLM genişletici (extender) | 0.51 ± 0.06 |

**Karar:** Tek yönlü bir "LLM işareti" olarak yorumlanmaz. Araç bunu rapor
alanı olarak değil, **bağlam** olarak kullanır. `LEXICAL_RICHNESS_DIRECTION`
sözlüğü dört rolün ölçülen değerini taşır ve rapor bunu gösterir; skorlanmaz.
Bu ablation MTLD/HD-D/MATTR'yi "kuramsal olarak denk" gerekçesiyle dışladığı
için o üçü de ayrı sinyal sayılmaz.

**Ekleme:** MTLD/HD-D/MATTR'nin AI tespiti özelliği olarak ölçülmüş etkisi
2024–26 literatüründe bulunamadı (§5). Buna karşılık "predictability + lexical
diversity" ikilisi 1,8M makine üretiminde ortalama **AUC-ROC ≈ 0.97** veriyor
(arXiv:2603.18482) — leksikal zenginliğin gücü buradan geliyor.

### 1.11 Kelime sayısı karıştırıcıdır

| Bulgu | Sayı | Kaynak |
|---|---|---|
| **Yalnız kelime sayısından** çapraz doğrulamalı ROC-AUC | **0.68** | arXiv:2609.26687 (Kumar vd., HICSS 2027) |
| İnsan yazısı uzunluğu | 620 ± 198 kelime | aynı |
| AI destekli yazı uzunluğu | 515 ± 155 kelime | aynı |
| Fark testi | **t(89) = 6.46, p < .001**; 90 çiftin **72'sinde** insan metni daha uzun | aynı |
| Ortalama cümle uzunluğu farkı | **yok** (27.2 vs 27.3, p = .97) | aynı |
| Cümle uzunluğu değişkenliği tek başına | **şansa yakın** | aynı |
| Uzun belgede yapı sinyali tersine güçlenir | 1/3/5 üretilmiş paragrafta LF+Motifs F1 0.43 → 0.71 | arXiv:2402.10586 |

**Karar:** `LengthConfound` belge uzunluğunu üç rejime ayırır (`cok_kisa` <
200, `calisma_kosuluna_yakin` 200–800, `uzun` > 800) ve her biri için ne
anlama gelmeyeceğini yazar. **Cümle uzunluğu özellikleri ağırlıklandırılmaz**;
ölçüm bunu destekliyor.

### 1.12 Kalibrasyon, ayrımdan ayrıdır

Araç 0–1 ölçekte puanlar üretiyor; bu bir olasılık **değil**. Üç ölçülmüş
gerçek:

| Bulgu | Sayı | Kaynak |
|---|---|---|
| Sıcaklık ölçeklemesini kaldırmak | F1 **80.17 → 80.16**, ECE **0.06 → 0.12**, Brier 0.22 → 0.27 | arXiv:2510.00890 (Sci-SpanDet, KBS 334:115123) |
| Eşiğin kendisi F1'i korpusa göre sallar | τ ∈ [0.1, 0.5] aralığında F1 farkı **0.8'e kadar**; τ = 0.1–0.2 "kabul edilemez FPR" | arXiv:2606.04906 (AITDNA) |
| Literatürde AI katılımı etiketlemesi için uyum (κ/α) | **hiçbir çalışma raporlamıyor** (HART, AITDNA, LLM-DetectAIve, DETree) | §5 |

**Karar:** `services/reliability.py` — ECE, Brier, güvenilirlik kovaları ve tek
parametrelik sıcaklık ölçeklemesi. **30 örnekten az** girdide sayı üretmez,
"olasılık olarak okunamaz" der. `checker calibrate` bunu her çalıştırmada
gösterir; gerçek korpusta (6+3 cümle) doğru biçimde ölçülmedi.

**Neden tek parametre?** İzotonik uyum küçük bir korpusu zarif bir eğriyle
ezberler ve hiçbir yere taşınmaz. Sıcaklık yalnız güveni yeniden ölçekler;
korporadaki gerçek serbestlik bu kadarı.

### 1.13 Türkçe: dedektörlerin ölçülmüş başarısızlığı

Bu, tez yazımı odaklı bir araç için en önemli bulgudur ve **Türkçe akademik
metin üzerinde** ölçülmüştür.

**Altıntop (2026)**, *Akademik Yayınlarda Kullanılan Yapay Zekâ Detektörlerinin
Güvenilirliği Üzerine Bir İnceleme*, DOI **10.56493/nkusbmyo.1866431**. 8 dedektör,
**hiçbir aşamada AI kullanılmadan** yazılmış 5.715 kelime / 40.407 karakterlik
Türkçe bir akademik metin üzerinde:

| Dedektör | %100 insan metne verdiği karar |
|---|---|
| **Justdone** | **%89 AI** |
| **ZeroGPT** | ~%80 AI (75.29% "büyük olasılıkla AI" + 21.14% "kısmen AI") |
| **Sidekicker** | "yapay zekâ üretimi işaretleri" (yüksek) |
| **MyDetector** | "büyük olasılıkla %40 AI üretimi" |
| **TruthScan** | %40 AI |
| QuillBot AI / Smodin / Copyleaks | %0 AI |

Aynı metin intihal motorlarında **tutarlı**: Turnitin %9 (yalnız alıntı filtresi) /
%4 (alıntı + kaynakça hariç), İntihal.net %10 / %6. Yazarların sonucu: hatalar
dedektörlerdedir.

**Dil etkisi, çapraz dil aracı için ölümcül.** Aynı çalışma Derrida'nın
*Plato's Pharmacy* "4. PHARMAKON" sayfasını üç dilde tarıyor:

| Dedektör | TR (2012 çeviri) | FR (1972 orijinal) | EN (1981 çeviri) |
|---|---|---|---|
| **ZeroGPT** | **%73.25 AI** | **%0 AI** | %3 AI |
| Sidekicker | %92 AI | %72 AI | %94 AI |
| QuillBot AI | %0 AI | %0 AI | %1 AI |

**73 puanlık salınım yalnızca dilden.** Aynı çalışma 1776 ABD Bağımsızlık
Bildirgesi'nin bazı dedektörlerce **%99.99 AI** sayıldığını da aktarıyor.

> **Karar:** `AI_TURKISH_FPR` her raporla birlikte döner. Araç hiçbir
> dedektör puanını olduğu gibi sunmaz; kendi ölçülmüş işletim noktasını kullanır
> ve `checker calibrate` ile kullanıcının kendi korpusunda yeniden ölçtürür.

### 1.14 Türkçe dedektör AUROC/FPR: sayılar var, teze taşmıyor

**Renklier & Sarıtaş (2026)**, DOI **10.28948/ngumuh.1930411**: 3.000 insan +
9.000 AI Türkçe metin (6 LLM × 3 senaryo × 5 alan), 11.990 temiz örnek,
kaynak-gruplu 70/10/20 bölme.

| Model | AUC-ROC | F1 |
|---|---|---|
| BERTurk (önerilen) | **99.31** | 98.37 |
| XLM-RoBERTa-base | 99.34 | 97.63 |
| mBERT | 98.86 | 97.19 |
| TF-IDF + SVM | 97.06 | 93.80 |

5 katmanlı grup-duyarlı CV: AUC-ROC **99.66 ± 0.15**, F1 98.56 ± 0.24.
Özgüllük **94.16**, recall 98.67 → **FPR = %5.84** (2.398 test örneği).

**Ama teze en yakın iki alanda en düşük:**

| Dilim | Doğruluk |
|---|---|
| haber / ödev | 99.79 / 99.58 |
| **akademik** | **95.21** |
| **mevzuat-hukuk** | **94.99** |
| senaryo: **yeniden yazım (rewrite)** | **96.51** (tam üretim 100.00) |
| uzunluk: 30–100 kelime | **94.2** (300–600 kelime 96.8) |

> **Karar:** Bu sayılar **başka bir çalışmadan alınmaz**. Tez metni akademik
> alandadır (95.21) ve yeniden yazım %96.51 ile en zor senaryodur — tez yazımında
> en olası kaçınma yolu. Araca özgü ölçüm gerekir.

### 1.15 Türkçede ölçülmüş dilbilimsel bulgular

| Bulgu | Sayı | Kaynak |
|---|---|---|
| **TTR sınıfları ayırmaz** | insan **0.756**; LLM aralığı **0.709–0.808** (Claude ve DeepSeek insanlardan *daha zengin*) | DOI 10.28948/ngumuh.1930411 |
| Türkçe LLM metni **daha kısa** | insan 234.7 kelime; LLM 141.2–184.8 (**%21–40 kısa**) | aynı |
| İnsan Türkçe **en yüksek** alt-kelime parçalanması | 1.522 token/kelime — her LLM'nin üstünde | aynı |
| LLM yeniden yazımı kaynakla leksikal örtüşmeyi yok eder | TF-IDF kosinüs: rewrite **0.531**, completion 0.299 | aynı |
| Jeneratör değişince Türkçe çöker | alan içi F1 **0.96–1.00** → alan dışı sıfır-shot **0.33–0.42** | arXiv:2408.10724 |
| Türkçe çapraz dil intihal tespiti | **TÜM** PAN / SemEval-2024 / GenAI-2025 görevlerinde yok | §5 |
| Çeviri altında tespit çöker | kesinlik: çeviri yok **%80** → çeviri **%26.7** → çeviri+paraphrase **%16.7** | DOI 10.33806/ijaes1026 |

> **Kararlar:** (1) TTR/hapax **Türkçede** skorlanmaz — `LEXICAL_RICHNESS_TURKISH`
> tablosu raporla birlikte gelir. (2) Uzunluk karıştırıcısı Türkçede daha
> keskindir, `LengthConfound.note` bunu söyler. (3) Token/kelime oranı **ters**
> çalıştığı için kullanılmaz. (4) Rapor, intihal yüzdesinin verilen kaynak
> kümesine bağlı olduğunu ve çeviri/eylul alıntılarının kaçırıldığını söyler.

### 1.16 Türkçe normatif çerçeve ve ölçülmüş AI dağılımı

**YÖK, *Yükseköğretim Kurumları Bilimsel Araştırma ve Yayın Faaliyetlerinde
Üretken Yapay Zekâ Kullanımına Dair Etik Rehber*, Mayıs 2024, 20 sayfa.**
Tam metin okundu. Üçü tez için önemli:

1. **Hiçbir sayısal eşik yok.** *benzerlik* ve *oran* kelimeleri **0 kez**,
   *yüzde* bir kez (eşik dışı bağlamda) geçiyor.
2. **Tezden hiç söz etmiyor.** Dört tesadüfi geçiş, hepsi genel. Rehber
   araştırma ve yayın faaliyetleri içindir; tez teslimine bağlayan bir hüküm
   değildir.
3. **İzinli/yasak ayrımı açık.** İzinli: hipotez, yöntem, örneklem büyüklüğü,
   güç analizi, veri analizi, veri toplama/saklama/paylaşma, **kaynak araştırması,
   kaynak düzenleme, dil bilgisi denetimi ve çeviri**. Yasak: **hipotez üretimi,
   tartışma, yorum ve uygulama** — "üst düzey beceri, deneyim ve uzmanlık
   gerektiren aşamalar". Koşul: araştırmacı çıktıyı gözden geçirip hataları
   düzeltmeli, tüm hukuki ve etik sorumluluğu üstlenmeli.

**Türkiye'de fiilen kullanılan benzerlik değerleri** (Altıntop 2026, Toprak 2017
ve Güçlüer vd. 2024'ten aktardığı; **bağlayıcı ulusal eşik değildir**):
genel beklenti **%15**, Türkiye'de "çoğunlukla" kabul edilen **%20**, tek
kaynaktan **≥%5** sorun sayılıyor.

**Ölçülmüş Türkçe AI oranı dağılımı** — Akkaya & Beygirci 2026,
DOI **10.46452/baksoder.1899625**: DergiPark'taki 68 üniversite dergisinden
**204** Türkçe makale, 2025, tamamı elle doğrulandı.

| AI oranı bandı | makale | pay | bant ort. |
|---|---:|---:|---:|
| 0–20% | **122** | **%59.8** | **%6** |
| 21–40% | 45 | %22.1 | %28 |
| 41–60% | 25 | %12.3 | %50 |
| 61–80% | 11 | %5.4 | %69 |
| 81–100% | 1 | %0.5 | %94 |
| **toplam** | **204** | | **ort. %20** |

Kontrol grupları: 20 AI-üretimi belge **%100** işaretlendi ("genellikle %40-60");
2010–2020 Türkçe makaleleri **%0–10**. TR Dizin (n=102) ort. **%12**, TR Dizin
dışı (n=102) ort. **%28**.

**Bölümlere göre:** giriş + literatür **100/204 (%49.0)**, bulguların yorumu 94
(%46.1), özet 78 (%38.2), sonuç 77 (%37.7), **yöntem 6 (%2.9)**.

> **Karar:** `TurkishAiPresence` raporla birlikte gelir. Türkçe verinin **%59.8'i
> %20'nin altında** ve o bandın ortalaması yalnız **%6** — bu araç %20'yi
> raporlama tabanı olarak kullanıyor, Türkçe veriden **bağımsız olarak** aynı
> eşik çıkıyor. İkinci okuma: AI tezde **girişe** yoğunlaşıyor, yönteme değil.
> Yöntemi temiz olan tezin girişi sorunluysa bu olağandır, şüpheli değil.

### 1.17 Ölçülebilir doğrulama: `scripts/measure_ratio.py`

Bu aracın kendi çiftleriyle yapılan ölçüm (dil başına 3 AI-benzeri + 3 insan
paragraf, `scripts/calibrate.py` örnekleri):

| Dil | Gözlemci / icra | Örneklem | İnsan medyanı | AI-benzeri medyanı | AUC | İnsan FPR |
|---|---|---|---|---|---|---|
| TR | `ytu-ce-cosmos/turkish-gpt2` / `hakanbogan/gpt2-turkish-cased` | 6 insan + 3 AI cümle (`checker calibrate`) | 0.359 (p10–p90 0.335–0.399) | 0.257 | **0.944** | **%0** (merkez 0.33) |
| TR | aynı çift | 2 insan + 4 AI cümle (`measure_ratio.py`) | 0.345 | 0.320 | 0.750 | %0 |
| EN | `gpt2` / `gpt2-large` (yayımlanmış çift) | 5 insan + 4 AI cümle | 1.129 (min 1.050) | 1.075 (min 1.033) | 0.800 | **%0** (yayımlanmış eşik 0.901) |
| EN | `gpt2` / `distilgpt2` (yayımlanmamış çift) | aynı | — | — | **0.250** | — |

İngilizce için yayımlanmış eşik 0.901, iki dağılımın da altında kalıyor: bu
temkinli tarafta (suçlama değil, atlama) bir tercih. Örneklemimize göre en iyi
eşik ~1.10; `checker calibrate` onu kendi korpusunda bulur.

Örneklemler küçük (cümle başına 3–6). Bunlar bir doğruluk iddiası değil,
**yönün ve ölçeğin** ölçümüdür: yön doğru, FPR %0, ayrım AUC 0.75–0.94 aralığında.
Kendi korpusunuzda `checker calibrate` çalıştırın.

İki tasarım hatası bu ölçümle bulundu ve düzeltildi:

1. **Her model kendi tokenizer'ıyla sayılıyordu.** Bu, "oran"ı tokenizer
   farkını ölçtü: TR AUC **0.375**. Düzeltme: gözlemcinin token dizisi iki modele
   de veriliyor (`score_token_ids`) → AUC **0.750**, sonra genişletilmiş örneklemle
   **0.944**.
2. **İşaret varsayımı.** Yayımlanmış çift (gpt2/gpt2-large) dışındaki çiftlerde
   oranın yönü ters olabilir: `gpt2`/`distilgpt2` ölçümünde AUC **0.250**. Bu
   yüzden `checker calibrate` yönü de ölçer ve ters çıkarsa uyarır; sabit yön
   varsayılmaz.

Türkçe için literatürde ölçüm olmadığı için `CHECKER_SCORE_RATIO_CENTER`
varsayılanları **kendi ölçümümüzdür** (TR 0.33) ve `checker calibrate` ile kendi
koleksiyonunuzda yeniden ölçülmelidir.

### 1.19 İngilizce: taban Türkçe'nin üçte biri

| Corpus | Ortalama | SD | Kaynak |
|---|---|---|---|
| Türkçe tezler (n=600, eğitim bilimleri) | %28.7 | 11.58 | Toprak 2014 |
| **İngilizce doktora tezleri (n=360)** | **%9** | **6** | Mayes 2017 |
| İngilizce makaleler (n=360, aynı alan) | %11 | 10 | aynı |

İngilizce doktora tezlerinin yalnız **%2.2'si** %25 üstünde. Yüksek puanın tek
anlamlı belirleyicileri **kaynakça sayısı ve kelime sayısı** (Nagelkerke
R² = 0.169) — uzun bibliyografya puanı şişiriyor, yazıyla ilgisi olan bir şey
değil. Referans listelerini dışarıda bırakmak puanları **anlamlı biçimde**
değiştirdi.

**Ölçülmüş eşik, politik tavan değildir:**

| | Değer | Kanıt |
|---|---|---|
| **Ölçülmüş optimal eşik** | **%15** | duyarlılık **%84.8**, özgüllük **%80.5**, AUC **0.902** |
| Kurumsal tavan | %25–30 | gözlenen normal değil, politik sınır |

Higgins, Lin & Evans (2016), DOI 10.1186/s41073-016-0021-8: 400 el ile doğrulanmış
manuskript, intihal **%25.8** (SD 9.9, n=66), temiz **%11.5** (SD 6.2, n=333).
Yalnız özet/giriş/sonuç/tartışma: **%25.7** vs **%5.6**.

> **Karar:** `EnglishBaseline` belge diline göre seçilir. `checker_similarity`
> `applied_baseline` alanı hangisinin kullanıldığını açıkça bildirir.

### 1.20 İngilizce kurum eşikleri ve eşiği reddeden kurumlar

Her URL **çekilerek** doğrulandı:

| Kurum | Kural |
|---|---|
| Virginia Tech, Graduate School | <%25 genel, tek kaynak <%5. **NOT:** aynı üniversitenin Honor System sayfası %15 de yayımlıyor — iki canlı sayfa çelişiyor, tek sayı olarak alıntılanmamalı |
| East Anglia (UK) | hacme göre: <%5 düşük, %5-20 orta, >%20 yüksek |
| Jinnah University (Pakistan, HEC) | toplam <%20, tek kaynak <%5; **kendi yayınları hariç** |
| Charles Sturt (Avustralya) | "genel olarak %25 altı güvenilir sayılabilir"; kurumun kendi örneği: doğru alıntılanmış bir deneme **%30** çıktı, intihal **yoktu** |
| Panjab University (Hindistan) | **bölüm bazlı**: giriş %30, literatür **%50**, yöntem %25, sonuç %10, tartışma %10, tam tez %20 |
| Rhodes Üniversitesi | "öğretim üyesi benzerlik endeksi için kabul edilebilir yüzde belirleyemez... endeks intihal düzeyinin göstergesi değildir" |
| Glasgow Caledonian | "intihalin başladığı ve bittiği bir kesim noktası yoktur"; en küçük eşik 3'e çekilirse puanın şiştiğini uyarıyor |
| ANU | iThenticate raporu zorunlu, sayısal eşik yok |

> **Karar:** Eşiği **reddeden** kurumlar boş satır değil, kendi gerekçeleriyle
> raporlanır (`no_threshold_reason`) — "araştırdık ve bulamadık" değil, daha
> güçlü ve alıntılanabilir bir konum. Bantlar belge diline göre filtrelenir.

### 1.21 İngilizce AI yaygınlığı ve alan farkı

| Venue | Özet | Giriş | Kasım 2022 |
|---|---|---|---|
| arXiv Bilgisayar | **%22.5** (21.7–23.3) | %19.6 | %2.4 |
| arXiv Elektrik/Sistem | %18.0 | %18.4 | %2.9 |
| **arXiv Matematik** | **%7.7** | **%4.1** | %2.5 |
| Nature portföyü | %8.9 | %9.4 | %3.4 |

1.121.912 makale; DOI 10.1038/s41562-025-02273-8. Tahmin yönteminin hata payı
**<3.5 puan**. Bağımsız doğrulama: Kobak vd. PubMed 2024 için **≥%13.5**
(DOI 10.1126/sciadv.adt3813); en iyi tek kelime 5.2 puan kaydırıyor. Bölüm
**yönü** yayımlanmış (özet/giriş/ilgili çalışmalar/sonuç > yöntem/deney),
sayıları UNVERIFIED. Beyan oranı: 200 incelenen makalenin **2'si** (%1.0). Hakem
yorumlarında LLM: %6.5–16.9 (arXiv:2403.07183).

> **Karar:** Matematik disiplininin **kendi önceliği** vardır (%7.7 / %4.1); CS
> oranıyla okumak önseli ters çevirir. `EnglishAiPresence` alan tablosunu taşır.

### 1.22 İngilizce sözlük sinyalleri (ölçülmüş, model gerektirmez)

| Bulgu | Sayı | Kaynak |
|---|---|---|
| LLM ile ilişkili aşırı kelimeler | 291 kelimelik nadir set + 10 kelimelik yaygın set; en iyi tek kelime **5.2 puan** | DOI 10.1126/sciadv.adt3813 |
| **LLM'ler `thus` ve `moreover`'dan kaçınıyor** | 1.25M tam metin (MDPI 2021–2025) | arXiv:2604.07565 |
| Terim büyümesi 2022→2024 | "delve" **+%1.500**, "underscore" +%1.000, "intricate" +%700 | DOI 10.1007/s11192-026-05601-5 |

> **Karar:** İki İngilizce sözlük kuralı eklendi (`llm_vocabulary`,
> `llm_avoided_connective`). **Türkçede karşılığı yok** — yayımlanmış bir Türkçe
> eşdeğer bulunmadığı için Türkçe modda bu kurallar **yok**. Asimetri kanıtın
> kendisinden geliyor, keyfî değil; bir test bunu açıkça sabitler.

### 1.23 İngilizce tür/uzunluk ve L2 uyarıları

| Bulgu | Sayı | Kaynak |
|---|---|---|
| Dedektör doğruluğu **türe** göre | Turnitin: insanlık **0.86** → fen **0.51** | DOI 10.1007/s40979-026-00213-1 |
| **Uzunluğa** göre | 300–330 kelime **0.87**; 450–550 **0.56** (χ²=8.41, p=.0149) | aynı |
| L2 yanlış pozitif (7 dedektör, TOEFL-91) | ortalama **%61.4**; Originality.AI **%75.8**, Quil.org %74.7 | DOI 10.1016/j.patter.2023.100779 |
| **Aynı veri setinde 2 yıl sonra** | **%23.1**; Çek non-native metininde entropi **yüksek** — yön tersine döndü | EACL 2026 SRW, DOI 10.18653/v1/2026.eacl-srw.20 |
| CEFR etkisi (Binoculars) | A2 **0.948** → B2 0.951 → dil bilgisi yok **0.898** | arXiv:2502.12611 |
| Doğrulanmış vakaların menşei | **%82'si** İngilizcenin resmî dil olmadığı ülkelerden | DOI 10.1186/s41073-016-0021-8 |
| İnsan L2 yazarın leksik çeşitliliği | CEFR A2 MTLD **58.49** (SD 22.68), A1 45.53 — L1 akademik medyan 93.6'nın **altında** | Shatz vd. 2026 |

> **Kararlar:** (1) `genre_and_length_note` her İngilizce raporla gelir: tek belge
> puanı yanlıştır. (2) `l1_l2_note` aynı dosyada. (3) **`replication_note`
> yönün değiştiğini yazar** — bu yüzden araç sabit bir yanlış pozitif oranı
> varsayamaz, yalnız kendi korpusunu ölçebilir.

### 1.24 İnsan İngilizce akademik metin referansları (kendi ölçümümüz)

Yayımlanmış bir tablo bulunmadığı için **4.462 pre-ChatGPT arXiv özeti**
(2015–2019, 742.788 token) ve **48 PMC OA tam metin** (113.928 token bölüm metni)
üzerinde ölçüldü. `PROVENANCE` alanı bunu açıkça "araştırma turunun kendi
ölçümü" diye işaretler; literatür değeri diye sunulmaz.

| Ölçüm | Değer |
|---|---|
| Cümle uzunluğu | 24.87 ± 5.62 |
| MTLD | 93.57 ± 31.87 |
| Leksik yoğunluk | %64.18 ± 4.45 |
| **Matematik MTLD** | **60.89** (diğer alanlar 94–102) |
| Bölüm MTLD medyanı | giriş 90.7 · **sonuç 54.6** · tartışma 85.2 |
| Bölüm cümle medyanı | giriş 27.3 · **yöntem 20.9** · sonuç 22.0 |

> **Kararlar:** (1) **Sonuç bölümü kurgusal olarak tekrarlıdır** — MTLD 54.6,
> tartışmanın 85.2'sinden düşük. Sonuçta düşük çeşitlilik zayıf kanıttır ve bu,
> Liang vd.'in "yöntem/deney en az etkilidir" bulgusuyla bağımsız olarak
> örtüşür. (2) **Matematik ayrı bir dildir** — İngilizce akademik metne göre
> kurulmuş her eşik onu yanlış işaretler; alan koşullandırma şart. (3) Yöntem
> bölümü en kısa cümlelere sahiptir ve `was/were/at/by` profili ayırt edicidir.

### 1.25 Çapraz dil: dil çifti değil, **kayıt türü** belirler

| Bulgu | Sayı | Kaynak |
|---|---|---|
| Wikipedia korpusunda çapraz dil (8 yöntemin karar ağacı **füzyonu**) | **%95.25 ± 1.76** | DOI 10.18653/v1/E17-2066 |
| **Bilimsel konferans korpusunda (TALN), aynı füzyon** | **%74.10 ± 1.29** | aynı |
| **TALN'deki en iyi TEK yöntem (CL-WESS)** | **34.49** | aynı |
| Çalışmanın kapsamı | **yalnız EN→FR**, 2017 gömme modelleri | aynı |
| PAN 2013'te tek çeviri adımı | PlagDet **11–15 puan** düşüş | CLEF 2013 |
| Türkçe MKQA çapraz dil getirme | 25 dil ortalamasının **üstünde** (BGE-M3 69.6 vs 65.8) | arXiv:2407.19669 |
| **EN↔TR bitext (WMT16), doğru rakamlar** | multilingual-e5-large **99.43** · all-MiniLM-L6-v2 **6.78** | DOI 10.18653/v1/2025.findings-emnlp.471 |
| EN↔TR bitext, aynı boyutta alternatif | gte-multilingual-base 97.98 · multilingual-e5-small 97.46 (117.7M, 118 MB int8) | aynı |
| EN↔FA paraphrase, 1.000 bilimsel özet | **düşüş yok**: Semi-Exact F1 0.97, Exact 0.96 | DOI 10.1371/journal.pone.0354459 |

> **Düzeltme (5. tur).** Bu bölüm önce iki yanlış rakam içeriyordu ve ikisi de
> düzeltildi: (a) *"%80 → %26.7 → %16.7 çeviri altında çöküyor"* genel bir sonuç
> gibi sunulmuştu; o sıra **tek** bir Arap edebiyat çevirisinde (Daly Walker,
> *I am the Grass*) üç zayıf sistemle ölçülmüş ve her sayı farklı içerik türü
> için **en iyi** sistemdir, genel doğruluk %23–26. Daha büyük ve modern bir
> ölçüm (EN↔FA, 1.000 özet) paraphrase'ta **hiç düşüş göstermiyor**. (b)
> TR-MTEB'de *37.02 / 73.07* rakamları `Mean(Task)` sütunuydu, bitext değil;
> doğru rakamlar **6.78 / 99.43** — yani İngilizce-only model ailesi
> Türkçe-İngilizce eşleştirmede **100 üzerinden 6.78**, yani hiç sinyal yok.
>
> **Karar:** İngilizce akademik kaynakta beklenti tek bir yöntemin %74'ü değil,
> **füzyonun** %74'üdür; doğrulama kümesi Wikipedia değil bilimsel kayıt olmalıdır.
> Gömmeye geçilirse tek aday `intfloat/multilingual-e5-small` (117.7M, 118 MB
> int8, EN↔TR 97.46); all-MiniLM ailesi **elenir** (6.78).

### 2.1b Çapraz dil: çeviriyle gelen aktarımlar

Kelime eşleştirmesi çevrilmiş bir aktarımı **%0** olarak raparlar; bu bir körlük
noktasıdır. Ölçülen tavan:

| Bulgu | Sayı | Kaynak |
|---|---|---|
| PAN 2013 tek çeviri adımı | PlagDet **11–15 puan** düşüş | CLEF 2013 |
| **Kayıt türü** tavanı belirliyor: Wikipedia / bilimsel konferans | **%95.25 ± 1.76** / **%74.10 ± 1.29** (8 yöntem füzyonu; en iyi tek yöntem TALN'de 34.49) | DOI 10.18653/v1/E17-2066 |
| EN↔TR bitext (WMT16) | multilingual-e5-large **99.43**; all-MiniLM-L6-v2 **6.78** | DOI 10.18653/v1/2025.findings-emnlp.471 |
| TR kaynak → LLM rewrite leksikal örtüşme | TF-IDF kosinüs **0.531** (rewrite), 0.299 (completion) | DOI 10.28948/ngumuh.1930411 |
| EN↔FA paraphrase (1.000 bilimsel özet) | **düşüş yok**: 0.97'e karşı 0.96 | DOI 10.1371/journal.pone.0354459 |
| Tek Arap çevirisi + 3 zayıf sistem | kesinlik %80 / %26.7 / %16.7, **genel doğruluk %23–26** — n≈1, zayıf kanıt | DOI 10.33806/ijaes1026 |

> **Karar:** `services/crosslingual.py` — gömme yerine **çeviri dayanıklı çapa**.
> Sayılar, ondalık hassasiyet, kimlikler, Latin alfabesi özel adlar, yazar–yıl
> çiftleri. Küme koşulu: ≥3 çapa, **aynı sırada**, 90 token içinde, en az 2'si
> sayı/kimlik. HyPlag'ın greedy tiling'i başka alfabeye uygulanmış hâli (MRR
> 0.79 vs 0.58).
>
> **Abartmama:** bir kontrol metni de küme üretir (alan içinde tekrar eden
> örneklem büyüklükleri). Bu yüzden `high | medium | review` dereceleri var ve
> okuma metni "kanıt değil işarettir" diyor. Tavan (%74.10) ve "bulunamayan küme
> hiçbir şey kanıtlamaz" uyarısı her raporla birlikte gider.

---

## 2. İntihal

### 2.1 n-gram/shingle: yüksek duyarlılık, düşük geri çağrım

| Bulgu | Sayı | Kaynak |
|---|---|---|
| PAN 2012 leksikal taban (birebir) | plagdet 0.50, geri çağrım 0.35, kesinlik 0.99 | Greiner-Petter vd., PAN 2025, arXiv:2510.06805 |
| Aynı taban, **LLM ile paraphrase** edilmiş kümede | plagdet **0.23**, geri çağrım **0.08**, kesinlik 0.59 | aynı |
| En iyi sistemler | PAN 2025'te 0.64 (gömme), PAN 2012'de 0.50 — sıralama ters dönüyor | aynı |
| Granularity cezası | `plagdet = F1 / log2(1+granularity)`; leksikal taban 2.20, en iyi sistemler 1.00–1.15 | aynı |

**Kararlarımız**

* Shingle **yüksek duyarlıklı ön eleme** olarak konumlandırıldı; raporda bu
  sınır yazılıyor. Recall'ın paraphrase'ta çöktüğü gizlenmiyor.
* `granularity` (vaka başına eşleşme sayısı) raporlanıyor; vaka sayımı kaynak
  değil **belge** tarafında kümeleniyor, çünkü üç kaynakta bulunan tek bir kopya
  üç vaka değil bir vakadır.
* `min_match_words` kurumsal filtreden geliyor (5 kelime, İstanbul Üniv. kuralı).

### 2.2 Yüksek eşleşme sayısı suç değildir

| Bulgu | Sayı | Kaynak |
|---|---|---|
| Eşleşen makale çiftlerinde atıf bağlantısı | **%6.4** (insan doğrulaması %7.0, CI 3.5–10.6) | Doron-Arad & Mossel, arXiv:2609.32963 |
| 2015→2025: eşleşen çiftler 449→1.831 (**4×**), atıflı çiftler 59→59 (**değişmedi**) | aynı | aynı |
| Gömme tabanlı kural | cos ≥ 0.84 **+** ≥6 ortak terim **+** terim örtüşmesi ≥ 0.10 **+** tam 5-token dizisi yok → kesinlik **%99.6**, geri çağrım %81.1 | aynı |
| PAN 2025 "altered" kontrolü (özlü ama kopyasız) | katılımcı sistemlerde geri çağrım %20'ye kadar düşük; iki taban **ters** döndü (Llama için ~2×) | arXiv:2510.06805 |

**Kararlarımız (tez için en önemli kısım)**

1. **Turnitin'in dört grubu** uygulanıyor: `cited_and_quoted` /
   `missing_quotation` / `missing_citation` / `not_cited_or_quoted`.
   Meşru alıntının cümle skoru 0.10 katsayıyla çarpılır; atıfsız olan 1.0.
2. Kaynakça/teşekkür/ön bölüm **sayımdan çıkarılır** — literatürde aynı malzeme
   temizleniyor ve bizim ölçümümüzde bu bölümler toplam kelimenin %8'i kadar
   gerçek "eşleşme" üretiyordu.
3. **Uydurma eşik yok.** Rapor, ölçülmüş Türkçe tez dağılımıyla karşılaştırılır:
   ortalama %28.7 ± 11.58 (n=600, eğitim bilimleri; TR dili %29.31, EN dili
   %24.37; Toprak 2014).

### 2.2b Yamalama (patchwriting): yüzde "nasıl" sorusunu yanıtlamaz

Yüzde "ne kadar" der. Tezin klasik riski ise başka: her şeyi yerinde meşru,
genelde birleştirilmiş yazmak. Nicelik-duyarlı ölçüt ailesi bunun için var ve
gerçek karşılığı HyPlag'dır (Meuschke vd., JCDL 2019, arXiv:1906.11761).

| Bulgu | Sayı | Kaynak |
|---|---|---|
| **GIT** (Greedy Identifier Tiles, matematik) — 10 doğrulanmış geri çekilen makale | MRR **0.79** | arXiv:1906.11761 |
| Aynı iş için sırasız `Histo` | MRR **0.58** | aynı |
| **Şans tabanları, 1.000.000 rastgele belge çifti** (ortak yazar/eş atıf çiftleri hariç) | GIT ≥ **0.15** · GCT ≥ 0.10 · Encoplot ≥ 0.06 · BC ≥ 0.13 · LCCS ≥ 0.22 · Histo ≥ 0.56 · LCIS ≥ 0.76 | aynı |
| Aday geri çağırma, üçünün birleşimi | 1.0 | aynı |
| Aynı yöntemin PAN-2025 → PAN-2012'de çöküşü | plagdet **0.61 → 0.17**; granülerlik 1.15 → 7.99 | arXiv:2510.06805 |
| Makine yeniden ifadesini insanlar bulma doğruluğu | **%53** (105 katılımcı) | 10.18653/v1/2022.emnlp-main.62 |

> **PPBS / "patchwork plagiarism score" diye bir metrik yok.** Hakemli
> literatürde izlenebilir bir tanım, aralık veya AUC bulunamadı; aramalar
> ilgisiz fizik/biyoloji makalelerine veya satıcı pazarlamasına çözülüyor.
> Brief'teki bu varsayım **yanlış**; gerçek ve ölçülmüş karşılık GIT'tir ve
> `services/patchwork.py` onu uygular.

**Kararlarımız**

1. **Skor = ||karolar|| / (I_d − 1)**, HyPlag formülünün birebir kelime
   karşılığı: ≥5 kelimelik eşleşmeler uzundan kısaya greedy kabul edilir, üst
   üste binenler atılır. Aynı paragrafın shingle + near + template
   tarafından üç kez raporlanması **bir** karo sayılır.
2. **Şans tabanı raporla birlikte verilir** (0.15). Literatürde türetilmiş tek
   eşik budur; her kurumsal yüzde yerel bir seçimdir ve türetilmiş bir
   dayanağı yoktur.
3. **Biçim ayrıca okunur.** COPE'nin yoğunlaşma ilkesi: birçok kısa ifadeye
   yayılmış tekrar, birkaç paragrafa yoğunlaşmış tekrardan **daha az**
   endişe vericidir. Karoların ortalaması 20 kelimenin altındaysa "yayılmış",
   üstündeyse "yoğunlaşmış" denir; 3'ten az karo varsa biçim okunmaz
   (dağılım yoktur).
4. **Paraphrase körlüğü her raporun içinde yazılıdır**, README'de değil.

### 2.2c Uydurma kaynakça: bu alandaki en iyi kanıtlanmış sorun

| Bulgu | Sayı | Kaynak |
|---|---|---|
| **Uydurma kaynak, biyomedikal** | 2,5M makale / 126M yapılandırılmış kayıktan **4.046 uydurma kaynak, 2.810 makale** (97,1M doğrulanmışın içinde) | *The Lancet* 2026, DOI 10.1016/S0140-6736(26)00603-3 |
| Oranın üç yıllık seyri | 2023: 1/2.828 → 2025: 1/458 → 2026'nın ilk 7 haftası: **1/277**. **>12 kat** | aynı |
| Tespit sistemi kesinliği | %91 | aynı |
| Etkilenen makalelerin dağılımı | **%91'i yalnızca 1–2** uydurma kaynak taşıyor; 246 makalede ≥3 | aynı |
| Hakemli bildiri (derleme) | %10.000 makalede 16.7 vs 10.6 (**%57 daha yüksek**) | aynı |
| ACL/NAACL/EMNLP 2024–25 | ~300 makalede ≥1 halüsinasyon kaynak; EMNLP 2025'in yarısı tek başına | arXiv:2601.18724 |
| ICLR/ICML/NeurIPS/USENIX Sec. | referans düzeyi oran <%1 ama 2025 NeurIPS/USENIX Security makalelerinin **~1/20'sinde ≥2** | arXiv:2607.00738 |
| HPC kongreleri | 2021: **%0** → 2025'te her kongrede **%2–6**; hiçbir yazar AI beyanı yok | arXiv:2602.05867 |
| Doğa ölçümü (2025) | 2025 bilgisayar bilimleri makalelerinin **%2,6**'sında şüpheli kaynak (2024: ~%0,3) | DOI 10.1038/d41586-026-00969-z |
| Ticari araçların başarısızlığı | Turnitin **%0**, OpenScholar **%0** (doğrulanmış vakalar) | arXiv:2502.16487 |
| LLM üretimi referanslarda öncül oran (biyomedikal) | **%30–69** | aynı |
| Aynı puanlama çubuğu: insan yazısı vs LLM önerisi | PeerRead insan makaleleri %0.8–6.3 → LLM önerileri **%24.0** (doğrulanmış), %36.0 (doğrulanamayan dahil) | aynı |
| Jüri düzeltmesi için gerekli doğrulayıcı kanıt | halüsinasyonlu/eksik kaynaklar açıkça sayılıyor | Adelphi Üniversitesi, 2025 |

**Kararlarımız**

1. `services/references.py` **yalnızca yapısal** denetim yapar: kalıcı kimlik
   (DOI/arXiv/URL), yıl makullüğü, yazar biçimi, yayın yeri, bozuk kimlik.
   DOI çözülmez, CrossRef sorgulanmaz — çünkü bir çözümleyicinin başarısız
   olma biçimi **yanlış suçlamadır** ve ölçülmüştür (PAN-2025'te naif tabanlar
   gerçek metni uydurma intihalin ~2 katı işaretliyor).
2. **Tek eksik alan işaretlenmez bile.** DOI'siz basılı Türkçe kaynak meşrudur.
   Yükseltme yalnız **küme** ile olur (≥3 güçlü eksik ya da tek başına bozuk
   DOI / gelecek yıl / imkânsız eski yıl) — ki uydurmanın ölçülmüş biçimi de
   kümelenmedir (%91 vakada 1–2 kaynak).
3. **Tez oranı için yayımlanmış ölçüm yok.** Yukarıdaki tüm sayılar makale ve
   konferans kayıtlarıdır. Bu dosyada UNVERIFIED olarak işaretlenir.

### 2.3 Kod: AST ve ucuz normalizasyon

| Bulgu | Sayı | Kaynak |
|---|---|---|
| AST (syntax) vs n-gram | AUROC **0.893** vs 0.765 (ConPlag2) | Ebrahim & Joy, arXiv:2604.25778, LEARNER 2026 |
| Yorum/boşluk/import temizliği | CrystalBLEU ve FusionTop3, Dolos'u (0.864) geçti | aynı |
| MOSS üst üste binen satır | medyan 8–9 satır; **%97–99** eşleşme 30 satırın altında | Ye vd., arXiv:2610.00863 |
| 30 satır eşiği | gerçek dosyaların yalnızca %2.9–3.5'i eşleşmeyi koruyor | aynı |
| Ölü kod + ifade düzeni normalizasyonu | ayrım %80–99 (Cliff's δ 1.00) | Maisch vd., *Same Same But Different*, arXiv:2510.25057, ICSE 2026 |
| AI yeniden yazımı | hiçbir yöntem ayıramıyor (δ ≤ 0.09) | aynı |

**Kararlarımız:** isim-bağımsız AST normalizasyonu (zaten vardı, testle kilitli),
docstring yok sayılıyor, öznitelik adları korunuyor. Eşik `min_match_words`
üzerinden açılır; README'de "30 satır eşiği gerçek sinyallerin %97'sini siler"
uyarısı var.

### 2.4 Denklem: kıyas alanı neredeyse yok

| Bulgu | Sayı | Kaynak |
|---|---|---|
| En iyi matematik-içerik benzerliği | plagdet **0.16**; en iyi genel yöntem 0.06 | Satpute vd., arXiv:2401.16969 (2024) |
| En zor operatörde (Formula Manipulation) F1 | **0.02** | aynı |
| Gerçek vakalarda operatör dağılımı | Paraphrase %78.6, konu değişimi %55.3, gösterim %45.9 | aynı |
| Uzmanlar arası uyum | κ = 0.39 ("fair") | aynı |

**Karar:** LaTeX normalizasyonu ile eşleştirme yapıyoruz ve **karşılaştırılacak
rakip yok** diyoruz. Yayımlanmış çubuk 0.16'nın altındadır; bunu iddia değil,
ölçülebilir bir gerçek olarak not ediyoruz.

---

## 3. Türkiye bağlamı

| Kurum | Kural | Kaynak |
|---|---|---|
| YÖK (ulusal) | **Oran belirlemiyor.** Tez için intihal yazılımı raporu alınıp danışman + jüriye gönderilir | Lisansüstü Eğitim ve Öğretim Yönetmeliği md. 9/2, 21/2 (RG 20.04.2016/29690) |
| YÖK (ÜYZ etiği) | **Hiçbir sayısal eşik yok**, "tez" kelimesi geçmiyor; izinli/yasak kullanım listesi var | Etik Rehber, Mayıs 2024, 20 s. (§1.16) |
| YTÜ (Temiz Enerji / SBE / FBE) | Alıntılar hariç ≤%15 · alıntılar dahil ≤%20 · tek kaynak ≤%2 | tet.yildiz.edu.tr |
| Başkent Üniv. Enstitüler | ≤%20 · tek kaynak ≤%2 | baskent.edu.tr |
| İstanbul Üniv. Sağlık Bilimleri | ≤%20 (kaynakça hariç, alıntılar dahil, 5 kelimeden küçük eşleşme hariç) | İÜ SB rapor kılavuzu |
| ESÜ LEE | ≤%30 toplam, tek kaynak ≤%15 | lee.eskisehir.edu.tr |
| Fiilen uygulanan TR normu (bağlayıcı değil) | genel **%15** · "çoğunlukla" kabul edilen **%20** · tek kaynak **≥%5** sorun | Altıntop 2026, Toprak 2017 ve Güçlüer vd. 2024'ten |
| Çukurova BADI / Akdeniz SBE | >%30 yazılı açıklama ister; "≤%30 hukuken intihal yok demek değildir" | BADI / Akdeniz SBE |
| YÖK Üretken YZ Etik Rehberi | **Yüzde yok.** Kullanılan bölümde açıklama zorunlu; hipotez/ tartışma/ yorum aşamalarında kullanım yasak | YÖK, Mayıs 2024 |
| TÜBİTAK UYZ Rehberi | Niteliksel beyan eşiği ("önemli ölçüde kullanım"), yüzde yok | TÜBİTAK, Ocak 2026 |
| YÖK'ün tez/makale YZ mevzuatı | Kasım 2025'te vaat edildi, **yayımlanmadı** | YÖK Başkanı konuşmaları (04–07.11.2025) |
| Tezlerde AI tespitinin dayanağı | "AI dedektör sonuçları tek başına disiplin sürecinin gerekçesi olamaz" | İstanbul Aydın LEE, Temmuz 2026 |

**Kararlarımız**

* `checker similarity` bu kurumsal filtre setini **varsayılan** olarak uygular ve
  beş kurumun eşiğini kaynak URL'leriyle birlikte raporlar.
* `checker compliance` beyan, atıf durumu, AI yoğun bölümler, benzerlik eşiği ve
  kapsama oranını denetlenebilir bir kontrol listesi olarak verir.
* Hiçbir yerde "intihal var/yok" hükmü yok; rapor eşleşmeyi, yerini ve atıf
  durumunu verir.

### Ölçülmüş Türkçe tez benzerliği dağılımı

| Ölçüm | Değer |
|---|---|
| Ortalama benzerlik (600 tez) | **%28.7** (SS 11.58) |
| Yüksek intihal sınıfı | %34.5 |
| Yüksek lisans | %29.44 · Doktora %25.46 (p<.001) |
| Türkçe dilde tez | %29.31 · İngilizce dilde %24.37 |
| Kaynak | Toprak, *Türkiye'de Akademik Yazı: İntihal ve Özgünlük*, Boğaziçi Üniv. Eğitim Dergisi 34(2), 2014 |

Bu tek büyük ölçekli Türkçe çalışma **2014** ve eğitim bilimleri alanında;
disiplinsel kırılım için 2025–26 döneminde doğrulanmış Türkçe veri **yok**
(`DOĞRULANAMADI`). Araç bu yüzden eşiği değil, ölçümü gösterir.

---

## 4. Türkçe AI kullanımı bağlamı

| Ölçüm | Değer | Kaynak |
|---|---|---|
| Genel popülasyonda GenAI kullanımı | %19.2 | TÜİK, Yapay Zeka İstatistikleri 2025 |
| Yükseköğretim mezunlarında | %36.1 | aynı |
| Türkçe öğrenci/araştırma metninde AI tespiti | in-domain doğruluk %0.95–0.97, **out-of-domain/GPT-4 F1 %0.35–0.42** | Üyük vd. 2024; Er 2025 (doktora tezi) |
| Türkçe haberlerde LLM yeniden yazımı | ~%2.5 (3.500 haber, 2023–2026) | Özdemir, arXiv:2602.13504 |
| Türkçe akademik metin stili ölçümü | **Yok** | `DOĞRULANAMADI` |

**Karar:** Türkçe'de kullanılan sınıflandırıcı İngilizce eğitilmiştir ve
ölçümde her cümlede %0–1 verir; bu yüzden otomatik olarak düşürülür ve gerekçesi
rapora yazılır. Türkçe'de asıl taşıyıcı, yerel dile uygun modellerle hesaplanan
likelihood-ratio sinyalidir (AUC 0.944, kendi ölçümümüz).

---

## 5. Doğrulanamayanlar (iddia edilmiyor)

* ~~Türkçe için herhangi bir dedektörün yayımlanmış AUROC/FPR değeri.~~
  **ÇÜRÜTÜLDÜ** — bkz. §1.14. Doğru ifade: sayılar var, ama teze en yakın
  kayıtlarda en düşük.
* Disipline göre Türkçe **tez** benzerlik oranları (TÜBİTAK Tez Merkezi,
  YÖK Teftiş Kurulu kararları dahil — hepsi doğrulanamadı, §1.16).
* Türkçe tez/sözlü/ödev için tipik **kelime** sayısı (yalnız sayfa aralıkları var).
* EXPEDITED, Unigram ve kriptografik watermark'lar için 2025–26 sayıları.
* SemEval/PAN'da gömme ile n-gram'ın **çapraz dil** paraphrase'taki kazancı.
* BigCodeBench'in bir intihal tespiti benchmarkı olarak kullanımı (üretim
  benchmarkıdır; tespit çalışması değildir).
* 2025–26'da yayımlanmış denklem intihal yöntemi (alan boşluğu).
* **"PPBS" / patchwork plagiarism score.** Hakemli literatürde izlenebilir bir
  tanım, aralık veya AUC yok; aramalar ilgisiz fizik/biyoloji makalelerine ve
  satıcı pazarlamasına çözülüyor. Karşılığı olarak HyPlag GIT uygulanıyor
  (§2.2b). Brief'teki isim **kullanılmıyor**.
* **"LWAI"** (*Look Who Is AI? … Stylometry and Discourse Analysis*) — arXiv,
  ACL Anthology, Crossref, DBLP ve genel web aramasında bulunamadı. Yerine
  doğrulanmış söylem-yapı hattı kullanıldı (§1.9).
* **MTLD / HD-D / MATTR'nin AI tespiti özelliği olarak ölçülmüş etkisi.** 2024–26
  literatüründe bulunamadı; en kapsamlı ablation (284 özellik) bunları
  "kuramsal olarak denk" gerekçesiyle dışlıyor (arXiv:2606.04177 §A.3.4).
* **Alt cümlecik yoğunluğu** ve **birinci/ikinci şahıs zamir oranı** için
  ölçülmüş katkı.
* **AI katılımının etiketlenmesinde annotator uyumu (κ/α)** — HART, AITDNA,
  LLM-DetectAIve, DETree hiçbiri raporlamıyor. Akademik metinde insan-deneyimli
  uzmanlar TPR %92.7 (arXiv:2501.15654) ama bu sözleşme notu değil, metin
  etiketlemesidir.
* **Bölüm bazında "sağlıklı tez" benzerlik yüzdesi.** 2024–26'da hakemli bir
  veri seti yok. Teklif numarası (%10 / %10–25 / %25–40) **ticari içeriktir,
  kanıtsızdır ve kullanılmaz**. COPE'nin rehberi de nicel değil niteldir.
* **İşaretlenen tezlerin yüzdesi, yeniden teste alınma oranı, temyizde
  bozulan vaka oranı.** Hiçbir veri seti bulunamadı. Tek bulunan tek kurum
  anlık görüntüsü Illinois FY25: 19 temyiz, 16 aynen (%84.2), 3 değişken
  (%15.8), **0 bozulan (%0)** — n=19, tüm suç türleri, intihale özgü değil.
* **Tezlerde uydurma kaynak oranı.** Yukarıdaki tüm sayılar makale/konferans
  kayıtlarıdır (§2.2c).
* **20–200 sayfalık belgelerde doğruluk–sayfa sayısı eğrisi.** En uzun
  akademik yerelleştirme çalışması 512 token girdi kullanıyor
  (Sci-SpanDet, arXiv:2510.00890); en uzun söylem çalışması ~8K token
  (TenPageStories). Böyle bir eğri yayımlanmamış.

---

## 6. Bu aracın ölçülen kendi performansı

| Ölçüm | Değer | Nasıl |
|---|---|---|
| TR likelihood-ratio AUC | 0.944 | `scripts/measure_ratio.py --lang tr` |
| TR insan metni FPR (merkez 0.33) | %0 | aynı |
| TR perplexity kalibrasyonu | insan medyan log10 ≈ 1.55, AI-benzeri ≈ 1.38 | `scripts/calibrate.py` |
| EN perplexity kalibrasyonu | insan ≈ 1.69, AI-benzeri ≈ 1.39 | aynı |
| Kod AST testleri | 12 | `tests/test_code_ast.py` |
| Toplam test | 412, model indirmeden ~5 sn | `make test` |
| MCP uçtan uca | 11 araç gerçek stdio istemcisiyle | `make mcp-test` |
| Raporlama tabanı ölçümü (yalnız stilometri, modeller kapalı) | gerçekçi TR giriş metni AI payı **%0.0**; jenerik metin **%57.7** | `tests/test_mcp.py` |
| Kalibrasyon koruması, gerçek korpusta | 9 örnek (6 insan + 3 AI cümle) → **ölçülmedi**, "olasılık olarak okunamaz" | `checker calibrate --human-dir … --machine-dir …` |