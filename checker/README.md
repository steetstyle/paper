# checker

Metni **cümle cümle böler**, her cümlenin **karakter/satır/sayfa/bölüm konumunu**
bilir ve iki şeyi hesaplar:

1. **AI yazım riski** — dört sinyalin ağırlıklı birleşimi: iki modelin
   likelihood-ratio'su (Binoculars ailesi), perplexity/burstiness (GPTZero mantığı),
   açık kaynak AI sınıflandırıcı ve stilometri.
2. **İntihal** — kelime n-gram birebir eşleşmeleri, rakam katlanmış şablon tekrarı,
   yeniden yazım yakın eşleştirmeleri, Python **AST** ve LaTeX karşılaştırması;
   her eşleşmenin **atıf durumu** ayrıca raporlanır.

Ayrıca, tezler için kurumsal çıktılar: **Turnitin tarzı benzerlik yüzdesi**
(15/20/2 kurumsal filtreleriyle) ve **YÖK uyum kontrol listesi**.

> **Bu araç hüküm vermez, sıralama verir.** Tek bir cümleyi "yapay zekâ üretmiş"
> diye işaretlemek mümkün değildir. RAID ekibi dedektörlerin cezai bağlamda
> kullanılmasına açıkça itiraz ediyor; IEEE S&P 2026 ticari dedektörlerin akademik
> bağlamda FPR'inin **%0.05–68.6** arasında ölçüldüğünü raporluyor. Bölüm
> [Sınırlar](#sınırlar) okunmadan kullanmayın.

Her tasarım kararının dayanağı: **[LITERATURE.md](LITERATURE.md)** — sayı, ölçüm
ve kaynak. Orada doğrulanamayanlar da açıkça listeli.

---

## Kurulum

Kendi sanal ortamı **checker/.venv** içinde yaşar; paper-app'in ortamına
bağımlılığı yoktur.

```bash
cd checker
bash scripts/bootstrap.sh --dev        # CUDA torch (~5 GB)
bash scripts/bootstrap.sh --cpu --dev  # CPU-only torch (~1 GB)
```

Pinned bağımlılıklar `requirements.txt` (çalışma zamanı) ve
`requirements-dev.txt` (test/ruff/mypy) içinde. Alternatif olarak:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

Doğrulama:

```bash
.venv/bin/checker doctor
.venv/bin/python -m pytest -q      # 181 test, model indirmeden ~5 sn
```

---

## Hızlı başlangıç

```bash
# Tam tez taraması: AI riski + intihal + üç çıktı
checker scan tez.md --profile thesis --ref-dir ./kaynaklar \
  --json rapor.json --md rapor.md --annotate tez.annotated.md \
  --compliance uyum.md

# Kurumsal benzerlik yüzdesi (YTÜ/İÜ/Başkent filtreleriyle)
checker similarity tez.md --ref-dir ./kaynaklar --md benzerlik.md

# YÖK uyum kontrol listesi: beyan, atıf durumu, AI yoğun bölümler
checker compliance tez.md --ref-dir ./kaynaklar

# Yamalama biçimi: metin kaç parçadan derlenmiş, şans tabanının üstünde mi
checker patchwork tez.md --ref-dir ./kaynaklar

# Kaynakça denetimi: hangi girdiler elle aranmalı (model indirmez)
checker references tez.md

# Kendi insan yazımın üzerinde işletim noktasını ölç
checker calibrate --human-dir ./kendi-yazilarim --machine-dir ./ai-ornekleri

# Model indirmeden, yalnız bölütleme (ayar yapmak için)
checker segment tez.md

# İki sürüm arası cümle farkı (kendi kendine intihal)
checker compare v1.md v2.md
```

`make` hedefleri: `bootstrap` · `bootstrap-cpu` · `test` · `lint` · `fmt` ·
`typecheck` · `check` · `scan` · `similarity` · `compliance` · `calibrate-local`

---

## Raporlama tabanı: %20

Turnitin'in AIW-2 protokolü (Ağu. 2024) bu eşiği kullanır ve nedenini sayıyla
verir: **719.877** pre-2019 insan öğrenci metni üzerinde belge FPR **%0.51**,
cümle FPR **%0.33**, belge geri çağırma %91.18.

Bu yüzden pay altındaysa **belge düzeyinde AI puanı verilmez**; yalnız konumlu
cümle bulguları raporlanır. Rapor şunu açıkça yazar:

```
AI kapsamı %14.2 (312 kelime) — raporlama tabanının (%20) altında,
belge düzeyinde AI hükmü verilmiyor
```

`ai_share` **saf `ai_score`** eşiğiyle hesaplanır: birleşik risk intihal de
içerdiğinden, atıfsız eşleşme dolu bir belge kendiliğinden AI payı kazanamaz.

## Derece iddiası yok

Araç hiçbir yerde "metnin %X'i AI tarafından yazıldı" demez. Gerekçe ölçülmüş
(APT-Eval, arXiv:2502.15666):

| Koşul | GLTR'nin yanlış pozitif oranı |
|---|---|
| dokunulmamış insan metni | %6.83 |
| kelimelerin **%1'i** düzenlendi | **%26.85** |
| kozmetik GPT-4o cilası | **%40.87** |

Dedektörler düzeltme derecesini de ölçemez (RoBERTa "hafif"→"ağır" arasında
4.3 puan; Fast-DetectGPT ters). Bu uyarı JSON'da `degree_caveat`, MCP'de
`verdict.caveats.degree` ve `checker://disclaimer` kaynağında taşınır.

## Yamalama (patchwriting)

Yüzde "ne kadar" der, bu komut "nasıl derlenmiş" diye yanıtlar:

```bash
checker patchwork tez.md --ref-dir ./kaynaklar
```

HyPlag'ın *Greedy Identifier Tiles* mantığı (JCDL 2019, arXiv:1906.11761):
≥5 kelimelik eşleşmeler uzundan kısaya greedy kabul edilir, üst üste binenler
atılır, sayı belge kelime sayısına bölünür. Aynı hacim iki farklı biçimde
farklı puan alır:

| Biçim | Karo sayısı | Skor |
|---|---|---|
| 21 kelime, tek blok | 1 | 0.037 |
| 21 kelime, üç parça | 3 | 0.111 |

**Şans tabanı 0.15** — literatürde türetilmiş tek eşik. HyPlag bunu **1.000.000
rastgele belge çifti** üzerinde ölçtü (GIT ≥ 0.15, GCT ≥ 0.10, Encoplot ≥ 0.06).
Her kurumsal benzerlik yüzdesi yerel bir seçimdir; türetilmiş dayanağı yoktur.

Biçim okuması COPE'nin yoğunlaşma ilkesini uygular: birçok kısa ifadeye
yayılmış tekrar, birkaç paragrafa yoğunlaşmış tekrardan **daha az** endişe
vericidir.

> **"PPBS" diye bir metrik yok.** Hakemli literatürde izlenebilir tanım, aralık
> veya AUC bulunamadı. Gerçek karşılık HyPlag GIT'tir ve o uygulanıyor.

## Kaynakça denetimi

```bash
checker references tez.md
```

Uydurma kaynakça, bu alandaki en iyi kanıtlanmış sorun. The Lancet (2026,
DOI 10.1016/S0140-6736(26)00603-3) 2,5M makale / 126M yapılandırılmış kayıt
taramasında **4.046 uydurma kaynak, 2.810 makale** buldu ve oran üç yılda
**12 kat** arttı:

| Dönem | Oran |
|---|---|
| 2023 | 1 / 2.828 |
| 2025 | 1 / 458 |
| 2026'nın ilk 7 haftası | **1 / 277** |

Etkilenen makalelerin **%91'i yalnızca 1–2** uydurma kaynak taşıyor — "sadece
iki şüpheli görünüyor" rahatlatıcı değildir. Adelphi Üniversitesi'nin 2025
tavsiyesi de halüsinasyonlu/eksik kaynakları, dedektör puanı yanında aranan
**doğrulayıcı kanıt** olarak sayar.

Denetim **yalnızca yapısaldır**: kalıcı kimlik (DOI/arXiv/URL), yıl
makullüğü, yazar biçimi, yayın yeri, bozuk kimlik. DOI çözülmez, CrossRef
sorgulanmaz. Tek eksik alan işaretlenmez bile — DOI'siz basılı Türkçe kaynak
meşrudur; yükseltme yalnız **küme** ile olur.

**Rapor, işaret sayısının ne anlama gelmediğini de söyler.** HALLMARK'ın gerçek
dağılım taramasında (arXiv:2607.18360) %1–2 temel oranında **en iyi doğrulayıcı
bile yalnız %5–18 teşhis oranı** veriyor — gerçek bir bulgu başına **4–9 yanlış
alarm**. Referans başına ölçülen uydurma oranı ise %0.31–0.81 (Phantom
References, arXiv:2607.00738).

Daha rahatsız edici bulgu: RefChecker'ın ölçülen FPR'si **%50.7** — 71 gerçek
referansın 36'sı yanlış işaretlenmiş — ve Phantom References bunların çoğunun
*uydurma kaynak* olmadığını, **girdi çıkarımı** olduğunu açıkça söylüyor: bozuk
yazar dizisi, kırpılmış başlık, editörün yazar sayılması, iki gerçek makalenin
başlık çakışması. Bu modülün kendi girdi çıkarımı da kaba, dolayısıyla **her
işaret elle aranmalı**.

> Aynı çalışmanın en ucuz ve en değerli bulgusu: veritabanları sabitken
> "herhangi bir kaynakta bulunamadı → işaretle" kuralı FPR **0.729** üretirken,
> "**tüm** kaynaklarda bulunamadı → işaretle" kuralı FPR'yi **0.049**'a indirir —
> **~15 kat**, ek veri kaynağı olmadan, sıfır maliyetle. Bu yüzden bu modül
> "bulunamadı"yı hiçbir zaman tek başına gerekçe saymaz.

> Çözümleyicinin başarısız olma biçimi yanlış suçlamadır: PAN-2025'te naif
> tabanlar gerçek metni uydurulmuş intihalin ~2 katı oranında işaretledi.
> Tezler için ölçülmüş bir uydurma kaynak oranı **yayımlanmamıştır**.

## Bağlam: puan değil, üç uyarı

Aşağıdaki üç sayı **skorlanmaz**, raporlanır. Her biri, tek başına
aşırı okunmaya açık olduğu için.

### Kelime sayısı karıştırıcıdır

| Bulgu | Sayı | Kaynak |
|---|---|---|
| Yalnız kelime sayısından ROC-AUC | **0.68** | arXiv:2609.26687 |
| İnsan / AI destekli uzunluk | 620±198 / 515±155; **t(89)=6.46, p<.001** | aynı |
| Cümle uzunluğu farkı | **yok** (27.2 vs 27.3, p=.97) | aynı |

Bu yüzden belge `<200 / 200–800 / >800` kelime olarak üç rejime ayrılır ve her
birinin ne anlama gelmeyeceği yazılır. Cümle uzunluğu özellikleri **ağırlıklandırılmaz**.

### Leksikal zenginlik en güçlü alan — ama yönü kararsız

284 özellikli ablation'da (27 LLM, 10 alan) leksikal zenginliğin düşürülmesi
alan içi **13.1**, alan dışı **27.7** F1 puanı kaybettirir; üç özelliğin kendisi
alan dışında tüm 284 özelliği **14.29 F1** farkıyla geçer (arXiv:2606.04177).
TTR, hapax ratio ve lexical density hesaplanır.

Ancak yön sabit değildir — 64.304 belgede:

| Rol | Sözlü zenginlik |
|---|---|
| İnsan yazar | 0.59 ± 0.07 |
| **LLM cilacı** | **0.61 ± 0.07** (insanın **üstünde**) |
| LLM üretici | 0.52 ± 0.06 |
| LLM genişletici | 0.51 ± 0.06 |

Hafif AI düzenlemesi zenginliği **yukarı**, üretim aşağı çeker. Bu yüzden
tek eşik "makine yazdı" anlamına gelemez; araç profili **bağlam** olarak
gösterir ve skorlamaz. 200 tokenin altında hiç hesaplanmaz.

### Puanınız olasılık değil

`checker calibrate` artık **ECE, Brier, güvenilirlik kovaları ve sıcaklık
ölçeklemesi** de raporlar. Gerekçesi ölçülmüş:

- Sıcaklık ölçeklemesini kaldırmak F1'i **80.17 → 80.16** değiştirirken ECE'yi
  **0.06 → 0.12**'ye ikiye katlıyor (arXiv:2510.00890). Aynı ayrım, bozuk olasılık.
- Eşiğin kendisi F1'i korpusa göre **0.8** kadar sallıyor (τ ∈ [0.1, 0.5],
  arXiv:2606.04906).
- Literatürde AI katılımı etiketlemesi için **κ/α raporlayan çalışma yok**.

**30 örnekten az** girdide sayı üretilmez — "puanlar olasılık olarak okunamaz"
denir. Küçük bir korpusla şöyle davranır:

```
kalibrasyon: 9 örnek, gereken en az 30. Kalibrasyon hesaplanmadı;
puanlar olasılık olarak okunamaz.
```

Kalibrasyon **ayrımdan ayrıdır**: mükemmel bir AUC, 0.62'nin "%62 olasılık"
okunabileceği anlamına gelmez.

## Çapraz dil: çeviriyle gelen aktarımlar

Kelime eşleştirmesi, çevrilmiş bir aktarımı **%0 benzerlik** olarak raporlar.
Bu en kötü cevaptır: temize çeviriliyor gibi görünür ama aslında bir körlük
noktasıdır. Ölçülen: PAN 2013'te tek çeviri adımı PlagDet'i **11–15 puan**
düşürüyor; Türkçe kaynak ile LLM yeniden yazımı arasındaki TF-IDF kosinüsü
**0.531** (DOI 10.28948/ngumuh.1930411) — çeviri leksikal örtüşmeyi gerçekten
azaltır, kelime eşleştirmesinin görememesinin nedeni budur.

Çözüm gömme değil, **çevirinin bıraktıkları**:

| Çapa | Neden dayanıklı |
|---|---|
| Sayılar | örneklem büyüklüğü, eşik, katsayı — 412, 0.80, 23 aynı kalır |
| Ondalık hassasiyet | 0.80 ile 0.8 farklı çapadır |
| Kimlikler | DOI, arXiv kimliği, ölçek adları |
| **Latin alfabesi özel adlar** | Türkçe metin yabancı soyadını çevirmez: Smith, Cohen, Kaufman |
| Yazar–yıl çiftleri | Smith (2019), Cohen (1988) |

Tek çapa tesadüftür; **küme** kanıttır: en az 3 farklı çapa, her iki metinde de
**aynı sırada**, 90 tokenlık pencere içinde, ve bunların en az 2'si sayı ya da
kimlik olmalı. Sıra şartı iki ilgisiz cümlenin aynı sayıları taşımasını engeller.

Bu, HyPlag'ın greedy tiling fikrinin başka bir alfabeye uygulanışıdır: sırayla
gelen yeniden kullanılmış *birimleri* say, ortak kelimeleri değil — ki MRR'ı
sırasız ölçüme karşı **0.79'a 0.58** çıkaran yapı budur.

**Neden gömme yok:** İngilizce-only modeller Türkçede **hiç sinyal vermiyor** —
TR bitext'te (WMT16 EN-TR) all-MiniLM-L6-v2 **6.78**, multilingual-e5-large
**99.43** (TR-MTEB, DOI 10.18653/v1/2025.findings-emnlp.471; *37.02 ve 73.07
rakamları `Mean(Task)` sütunudur, bitext değil — önceki sürümde bu atıfı
kullanmıştık*). Gömmeye geçilirse tek aday `intfloat/multilingual-e5-small`
(117.7M parametre, 118 MB int8, EN↔TR **97.46**); all-MiniLM ailesi elenir.
Ayrıca çeviri gömmenin dayandığı şeyi yok ediyor (yukarıdaki kosinüs).

### Küme güven dereceleri — abartmamak için

Bir **kontrol** metni de küme üretir: tesadüfen "412 katılımcı, 2019, Cohen
(1988), 0.80" içeren, ilgisiz bir Türkçe metin. Alan içinde örneklem
büyüklükleri ve eşikler tekrarlanır, bu yüzden yalnız sayılardan oluşan kısa bir
küme tesadüfen oluşabilir. Bu yüzden:

| Seviye | Koşul |
|---|---|
| `high` | ≥6 çapa **ve** en az bir özel ad ya da DOI |
| `medium` | ≥6 çapa, ya da özel ad içeren daha küçük küme |
| `review` | yalnız sayılardan oluşan kısa küme — tek başına anlamsız |

Okuma metni **"kanıt değil işarettir"** der ve elle doğrulama ister. Ölçülen tavan
raporla birlikte gider: Wikipedia'da %95.25, **bilimsel konferans korpusunda
%74.10** (DOI 10.18653/v1/E17-2066) — ama bu rakam **sekiz yöntemin karar ağacı
füzyonudur** ve o korpusta en iyi **tek** yöntem **34.49**'dur; çalışma yalnız
EN→FR yönünde ve 2017 gömme modelleriyle yapılmıştır. Yani bulunan küme
doğrulamayı gerektiren bir işarettir; **bulunamayan küme hiçbir şey kanıtlamaz.**

## PDF: tezler nasıl geliyor, tez gibi geliyor

Tezler Markdown olarak gelmez. Bu bölüm, arXiv'deki **gerçek** bir doktora tezi
(1407.6566, 154 sayfa, 60.238 sözcük) üzerinde ölçülenlerdir — düzeltme
**öncesi** ile **sonrası** karşılaştırması.

Düz `extract_text()` yalnızca glifleri verir; punto bilgisi kaybolur, dolayısıyla
başlık paragraftan **ayırt edilemez**. Ölçülen sonuç:

| | önce | sonra |
|---|---:|---:|
| Bölüm | **1** | **84** |
| Başlık bloğu | 0 | 83 |
| Kaynakça denetimi | "bulunamadı" | 189 kayıt |
| Sayfa haritası | ✓ | ✓ |

**Nasıl:** pypdf'in ziyaretçi geri çağrısıyla karakter başına punto toplanır,
gövde punto karakter ağırlıklı mod ile belirlenir (bu tezde 10.9pt / 92.226
karakter; başlıklar 12pt ve üstü / 978 karakter), başlıklar Markdown `#`
önekine çevrilir ve mevcut bölücü **hiç değiştirilmeden** çalışır.

**Ölçülen doğruluk:** gövde üstündeki 46 parçanın **44'ü gerçek başlık**, 2'si
artefakt (arXiv damgası, dipnot yıldızı) — **%95.7 kesinlik**, iki artefakt
filtreyle gider.

### Neden düz metinden numaraya bakmak yetmiyor

İlk deneme "1.2.1 Optical observations" gibi satırları numaradan tanıdı ve
**359 sahte başlık** üretti: içindekiler tablosu (nokta kılavuzlu), sınav jürisi
listesi ("1. Prof. Dr. Steinmetz") ve "7.5 keV) for 345 systems..." ile başlayan
bir cümle. Doğru kural **punto başlığın mı olduğunu belirler**, numaralandırma
yalnız **seviyeyi**.

Sarma başlıklar da çözüldü: "The XMM-Newton/SDSS" ve "Galaxy Cluster Survey" aslında
tek başlıktı. Kural tipografiktir — **ardışık, aynı puntodaki satırlar** kaydırılmış
tek başlıktır; **punto değişimi** yeni başlık başlatır. Üç başlık ("Chapter 1"
20.7pt, "Introduction" 24.8pt, "1.1 Clusters" 14.3pt) aralarında boş satır olmadan
üst üste geldiği için boş satır sezgisi yetmiyordu.

### İki hata, ikisi de sessizce

1. **Sayfa altbilgisi başlığı yuttu.** Altbilgi numarası, sayfa sınırı (`\f`) ve
   başlık işareti tek satırda birleşiyordu: `126\f# References`. Başlık deseni
   satır başına çivili olduğu için artık eşleşmiyor — ve bu **tek satır**
   yüzünden 154 sayfalık tez "kaynakça yok" diyordu.
2. **Gövde punto sayfa başına hesaplanıyordu.** Bir tablo sayfasında mod tablo
   kendi fontu oluyor, sonra o sayfadaki sıradan metin başlık gibi görünüyordu:
   400 karakterlik bir tablo-notu paragrafı bölüm haline gelmişti. Gövde punto
   artık **belge genelinde** hesaplanıyor.

### Kaynakçada ölçülen yanlış pozitifler

Astronomi biçimi `Yazar. Yıl, Dergi, Cilt, Sayfa [geri referanslar]`. İki hata
vardı ve ikisi de gerçek ölçümle bulundu:

- `max(years)` **sayfa numaralarını yıl** sanıyordu: `MNRAS, 403, 2063 11` →
  "2063 gelecek yıl". Yıl artık **konumundan** okunuyor (yazarlar ile yayın yeri
  arasındaki, virgülle biten ilk dört haneli sayı).
- `MIN_PLAUSIBLE_YEAR = 1950` **temel fizik referanslarını** işaretliyordu:
  Hubble 1926, Zwicky 1933 ve 1937, Smith 1936. Dokuz "imkânsız eski yıl"
  işaretinin **tamamı meşru** çıktı. Eşik artık ilk bilimsel dergilerin
  başlangıcı; 1950 öncesi kabul edilir, çünkü **öyle olmak zorunda**.

Sonuç: 189 gerçek kaynakta **9 review → 2**, ikisi de çıkarım artefaktı.

### İkinci tur: iki tez daha, iki alan

Aynı ölçüm iki farklı tezde tekrarlanınca ikinci bir hata ortaya çıktı: **bütünlük
denetimleri bilimde ağlıyordu**.

| tez | alan | sözcük | bölüm | kaynakça | `tampered` |
|---|---|---:|---:|---:|---|
| 1407.6566 | Astrofizik (LaTeX) | 60.238 | 84 | ✓ | yanlış → **doğru** |
| 2306.14650 | YZ (Fransızca-Türkçe) | 42.094 | 56 | ✓ | yanlış → **doğru** |
| 1911.03731 | ML (İngilizce) | 45.889 | 72 | ✓ | yanlış → **doğru** |

`repeat_punctuation` sinyali **hiç ayırt etme gücü taşımıyordu**. Üç gerçek tezde
tüm eşleşmeler meşruydu:

- elips: `......., Takey`, `j1, j2, ....`
- ADS kodu: `A&A...534A..120T`
- **display-math**: `(x1, h'(x1)), ..., (xm, h'(xm))` ve matrisin dikey elipsi,
  ki metin çıkarıcı onu `... . . . ...` olarak yazıyor — ML tezinde **64 adet**
- fonksiyonel analiz: `∫[Dφ1...Dφn]`, `ξ(θ1...θn)`

Meşru belgeler 3, 5 ve 10 eşleşme üretti ve **kalibrasyon için tek bir gerçek
quote-manipülasyon örneği bulunamadı**. Bu yüzden gözlem raporda duruyor ama artık
tek başına 60.000 sözcüklük bir tezi "manipüle" sayıp AI skorlarını düşürmüyor.
Kalibre edilemeyen sinyal hüküm dönüştürmez — raporlama tabanının da izlediği kural.

Ayrıca `greek_notation` bulgusu **kaldırıldı**: ML tezinde 316 örnek (hν, λ, θ, α,
β, μ) veriyor, yani eyleme bağlı olmayan bir bulgu salt gürültü. Korunması gereken
şey zaten homoglif kuralının içinde: Yunan gösterim, Kiril saldırı vektörü.

Ve bozukluk mesajı artık **hangi bulgunun** tetiklediğini söylüyor: homoglif bir
*ikame*, görünmez karakter bir *ekleme* — ve RAID'in ölçtüğü %40.6 düşüş ikamesine
ait, eklemeye değil.

### Üçüncü tur: iki fizik tezi, ve sessizce atılan bir tez

**0911.2782** (string kuramı, 152 sayfa) ve **1912.04141** (yoğunlaştırılmış madde
fiziği, 191 sayfa). İkisi de farklı üniversite şablonlarından. Beş tezin tamamı:

| tez | alan | sayfa | sözcük | bölüm | kaynakça cümlesi |
|---|---|---:|---:|---:|---:|
| 1407.6566 | Astrofizik | 155 | 60.238 | 85 | 112 |
| 2306.14650 | YZ (FR/TR) | 153 | 42.094 | 56 | 770 |
| 1911.03731 | ML | 120 | 45.889 | 72 | 60 |
| 0911.2782 | String kuramı | 153 | 52.620 | 96 | 561 |
| 1912.04141 | Yoğunlaştırılmış madde | 192 | 78.422 | 91 | 930 |

#### 1. En ciddi bulgu: sessiz kırpma

`max_chars` **400.000**'dı. 191 sayfalık yoğunlaştırılmış madde tezi
**466.068** karakter — yani kırpılıyordu. Ve kırpılan yer belgenin **sonu**,
yani **kaynakça** idi. Sonuç: kaynakça denetimi **417 kayıt yerine 19** kayıt
gösteriyordu ve hiçbir yerde bunun sebebini söylemiyordu.

Sınır **2.000.000**'a çıkarıldı (ölçülen beş tez 240.834–466.068 karakter), ve
kırpma artık `degraded_signals` içinde **bildiriliyor**: kaç karakter atlandığı,
atlanan yerin belgenin sonu olduğu ve kaynakça denetiminin eksik kalabileceği.

#### 2. Gövde punto ile yazılmış kaynakça başlığı

0911.2782'de `References` başlığı **10.9pt** — yani tam olarak gövde punto. Onun
için punto analizi ulaşamıyor ve tezin **kaynakçası hiç denetlenmiyordu**.
Çözüm puntodan değil içerikten geliyor: **tek başına duran** bir kaynakça anahtar
kelimesi (`References`, `Bibliography`, `Kaynakça`, …) başlıktır, ne kadar puntoyla
yazıldığına bakılmaksızın.

#### 3. Kaynakça girişlerini bölen kurallar

Bu iki tezin kaynakça stilleri daha da farklıydı ve üç hata çıktı:

- **Kayıtlar kendini numaralandırıyordu** (`[1] …`, `[341] …`) ve yazar-bölme
  kuralı `[1] ` sonrasında kesiyordu: girişler ortadan başlıyor, bir kaydın kuyruğu
  sonrakinin başı oluyordu. Şimdi liste numarası doğrudan sınır olarak alınıyor.
- **DOI bir tarih değil.** `10.1103/PhysRevB.77.220503` içinde `1103` var;
  417 kaydın **263'ü** yılı bir DOI'den okuyordu. Kimlikler (DOI, URL, ISBN,
  arXiv) yıl aranmadan önce metinden çıkarılıyor.
- **Yıl deseni aslında üç haneydi.** `1[5-9]\d` deseni `1998` için `199`'u
  eşleştirip dördüncü hanede takılıyordu — yani **1950–1989 arası yıllar hiç
  aday olmuyordu.** Hubble 1926, Zwicky 1933/1937, Smith 1936 bu yüzden
  bulunamıyordu.

#### 4. Yıl hangi konumda? İki stil birbirine zıt

| stil | örnek | yıl |
|---|---|---|
| Astrofizik | `Berlind, A. A., et al. 2006, ApJS, 167, 1 4, 11` | **ilk** |
| hep-th | `vol. 3, pp. 1415–1443, 1999, hep-th/9811131` | **son** |
| APA | `Smith, J. ve Jones, P. (2019). …` | parantez içi |

İlki virgülle biten sayıyı alınca **cilt** okunuyordu (167, 172, 189), sonkini
alınca **sayfa aralığının** içinden bir sayfa okunuyordu (1443, 1453). Ayrım şu:
**yıldan sonra harf gelir** — yayın yeri ya da arXiv kimliği. Cilt ve sayfa
numarasından sonra ise sayı ya da hiçbir şey gelir.

Sonuç: astrofizik tezinde yanlış yıl **14 → 1**, yoğunlaştırılmış madde tezinde
**263 → 63**, `review` sayısı **344 → 158**.

### Dördüncü tur: iki tez, iki şablon ailesi

**1706.08318** (Barselona, kuantum bilgi — **Katalanca/Fransızca/İngilizce** karışımı,
Roma rakamlı bölümler) ve **2203.03469** (veritabanı, 220 sayfa — **harf aralıklı**
başlıklar). Yedi tezin tamamı:

| tez | alan | sayfa | sözcük | bölüm | kaynakça cümlesi |
|---|---|---:|---:|---:|---:|
| 1407.6566 | Astrofizik | 155 | 60.238 | 85 | 112 |
| 2306.14650 | YZ (FR/TR) | 153 | 42.094 | 56 | 770 |
| 1911.03731 | ML | 120 | 45.889 | 72 | 60 |
| 0911.2782 | String kuramı | 153 | 52.620 | 96 | 561 |
| 1912.04141 | Yoğunlaştırılmış madde | 192 | 78.422 | 91 | 930 |
| 1706.08318 | Kuantum bilgi (3 dil) | 187 | 57.481 | 132 | 418 |
| 2203.03469 | Veritabanı / SQL | 220 | 79.385 | 27 | 2.302 |

#### 48.7pt'lik bir filigran, başlıktan büyük

Bu tezin şablonunda sayfa başına bir-iki kez **48.7pt**lik süsleme harfleri var
(`S`, `T`, `F`). Numaralandırmada her gerçek başlıktan **büyük** oldukları için
en üst düzey bölümlere terfi ediyorlardı — 41 başlıktan 15'i sahteydi. Tek karakter
asla başlık değildir.

#### Harf aralıklı başlıklar

Bölüm başlıkları `I N T R O D U C T I O N` biçiminde — harf aralığı, içerik
akışında **gerçek boşluk** olarak duruyor. Bu iki şeye birden zarar veriyordu:
başlık metni okunmuyordu ve `1I` bitişik olduğu için numaralandırma kuralı da
göremiyordu.

Kritik bulgu: bu tezin kaynakça başlığı **`B I B L I O G R A P H Y`** ve
**gövde puntosunda**. Yani harf aralığı hem onu, hem gövde punto eşiğini
geçirdi. Sonuç: 220 sayfalık tezin **567 kayıtlık** kaynakçası **hiç** bulunamadı.
Anahtar kelime artık eşleştirme için birleştirilmiş metin üzerinden soruluyor ve
tek kelime olan başlıklar temiz formuyla yayımlanıyor.

**Ama çok kelimeli harf aralıklı başlıklar birleştirilmiyor.** pypdf karakter
başına değil **run başına** x konumu veriyor, yani boşluk genişlikleri kurtarılamıyor
ve kelime sınırları gerçekten kaybolmuş:
`I N T R O D U C T I O N A N D B A C K G R O U N D` üç kelime mi bir mi belirsiz.
Birleştirmek `INTRODUCTIONANDBACKGROUND` gibi **uydurulmuş** bir kelime üretir —
gizlediği artefaktten daha kötü. Görünür artefakt dürüsttür; yalnız numaralandırma
kuralı, `1I` yerine `1 I` görmek için birleştirilmiş formu kullanır.

Türkçe harfler de kapsama alındı: `K A Y N A K Ç A` de aynı artefakttır.

### Kalan sınırlar (ölçülmüş, gizlenmiyor)

- **2203.03469'da `Part I` / `Part II` bölüm başlıkları bulunamıyor** — 12pt'te
  ama takip eden harf aralıklı satırla aynı punto olduğu için tek bloğa giriyor.
  Tezin bölüm yapısı bu yüzden kısmi (27 bölüm).
- **Bir liste sayfasındaki girdiler başlık sanılıyor** (`BR E L I B R A R Y`,
  `CC F G R U L E S`) — sayı ya da noktalama eşleşmesi olmayan, puntoyla
  başlık ölçüsündeki liste satırları. Bunlar `review` olarak **işaretlenmiyor**,
  yalnızca bölüm tablosunda görünüyor.

### Beşinci tur: dört tez, 1996'dan bugüne

Dört matematik/fizik tezi: **math/0611002** (Imperial, K-stabilite),
**1702.04123** (Varşova, Gysin homomorfizması), **q-alg/9607022** (1996,
Habilitationsschrift) ve **1102.0985** (Kähler/Yang-Mills). On bir tez, sekiz
disiplin, 86–220 sayfa, **dokuz farklı şablon ailesi** — 1996 TeX'ten bugünkü
MS Word'a.

| tez | şablon | sayfa | sözcük | bölüm | kaynakça |
|---|---|---:|---:|---:|---:|
| 1407.6566 | LaTeX | 155 | 60.238 | 85 | 112 |
| 2306.14650 | LaTeX | 153 | 42.094 | 56 | 770 |
| 1911.03731 | LaTeX | 120 | 45.889 | 72 | 60 |
| 0911.2782 | LaTeX | 153 | 52.620 | 96 | 561 |
| 1912.04141 | LaTeX | 192 | 78.422 | 91 | 930 |
| 1706.08318 | LaTeX | 187 | 57.481 | 132 | 418 |
| 2203.03469 | **MS Word** | 220 | 79.385 | 27 | 2.302 |
| math/0611002 | 1990s LaTeX | 88 | 29.091 | 44 | 60 |
| 1702.04123 | LaTeX | 86 | 27.787 | 17 | 187 |
| q-alg/9607022 | **1996 TeX** | 103 | 35.373 | 29 | 211 |
| 1102.0985 | LaTeX | 121 | 54.432 | 13 | 100 |

#### Bir fiziksel operatör emoji sanıldı — 231 kez

1996 tezinde `emoji` sayısı **231**'di. İncelendiğinde hepsi aynı karakter:
**Klein-Gordon d'Alembertian işareti `□`**. TeX fontları bu operatörü
**Dingbats** bloğundaki (U+2700–U+27BF) bir koda yerleştiriyor — o blok süsleme
için ayrılmıştı, TeX oraya matematiği koydu.

Denklem `(□ + m²)φ = 0` — yani bölümün konusu olan denklem — 231 kez "emoji"
olarak işaretleniyordu. Menzil gerçek piktoğrafik bloklara daraltıldı
(U+1F300–U+1FAFF ve Miscellaneous Symbols). **Kayıp:** Dingbats'taki gerçek
emoji'ler (✅ gibi). Tezde emoji neredeyse hiç olmadığı ve bulgunun yalnızca
bilgilendirici olduğu için kabul edilebilir bir denge — ama bu bir ödünleşmedir,
kazanç değil.

#### Eski PDF'ler rakamları boşlukla yazıyor

1996 tezinde yıl **`(1 996)`** olarak geliyor. Sonuç: 85 kaydın 41'i
"yılsız" işaretlendi. Rakamlar arasındaki boşluk kapatılıp yeniden aranıyor —
güvenli, çünkü gerçek bir sayfa aralığı (`1415 1443`) birleşince sekiz haneli
oluyor ve hiçbir dört haneli desen ona uymuyor.

#### Sayfa sonunda bölünen kayıtlar

PDF metni sayfa arasında `\n\f\n` ile üretiliyor, yani araya **3 karakter**
giriyor; bitişiklik eşiği 2 idi. Sonuç: `B.W. Lee, in Methods in Field Theory,
ed. R.` ile `Delbourgo, D. Kreimer, Phys.Lett.B366 (1996) 421` — **tek bir
kayıt** — ikiye bölünüyor ve yıl atılan yarıda kalıyordu. Eşik 3'e çıkarıldı.

Astrofizik tezinde `review` **2 → 1**, string kuramında **7 → 6**, yoğunlaştırılmış
madde'de **158 → 142**.

## Söylem profili

Rapor, dört ilişki sınıfını (atıf / sıralama / açımlama / neden) TR/EN
bağlaçlarıyla sayar ve 100 kelimeye normalize eder. Yön ölçülmüştür: insan
yazarlar atıf ve sıralamayı, LLM'ler açımlamayı fazla kullanır (RACE,
arXiv:2604.04932). Önemli olan, bu yapının **düzenlemeden sonra da ayakta
kalması**: insan→AI cilalama kosinüs benzerliği 0.92 ± 0.08.

Tek başına zayıf (ölçülen motif F1'si 0.49–0.61), bu yüzden **hüküm üretmez**,
yalnız bağlam verir.

---

## İngilizce desteği

İngilizce modu ayrı tablolarla gelir, çünkü **ölçülen tabanlar 3 kat farklıdır**.
Türkçe tez tabanını İngilizce bir belgeye uygulamak, normal ve iyi kaynak gösterilmiş
bir tezin büyük kısmını işaretlemek olurdu.

| Corpus | Ortalama | SD | Kaynak |
|---|---|---|---|
| Türkçe tezler (n=600, eğitim bilimleri) | %28.7 | 11.58 | Toprak 2014 |
| **İngilizce doktora tezleri (n=360)** | **%9** | **6** | Mayes 2017 |
| İngilizce yayınlanmış makaleler (n=360) | %11 | 10 | aynı |

İngilizce doktora tezlerinin yalnız **%2.2'si** %25'in üstünde. Aynı çalışmada
yüksek puanın **tek anlamlı belirleyicileri kaynakça sayısı ve kelime sayısı**
(Nagelkerke R² = 0.169) — uzun bibliyografya puanı şişiriyor, yazıyla ilgisi
olan bir şey değil.

**Ölçülmüş eşik ≠ politik tavan:**

| | Değer | Kanıt |
|---|---|---|
| **Ölçülmüş optimal eşik** | **%15** | duyarlılık %84.8, özgüllük %80.5, AUC **0.902** (Higgins vd. 2016, 400 el yazımı doğrulanmış manuskript) |
| Kurumsal tavan | %25–30 | bir **politik** sınırdır, gözlenen normal değil |

Doğrulanmış intihal vakaları ortalama **%25.8** (SD 9.9), temiz olanlar **%11.5**
(SD 6.2). Yalnız özet/giriş/sonuç/tartışma bölümlerine bakınca ayrım açılıyor:
**25.7** vs **5.6**.

### Alan farkı

| Venue | Özet | Giriş | Kasım 2022 tabanı |
|---|---|---|---|
| arXiv Bilgisayar | **%22.5** | %19.6 | %2.4 |
| arXiv Elektrik/Sistem | %18.0 | %18.4 | %2.9 |
| **arXiv Matematik** | **%7.7** | **%4.1** | %2.5 |
| Nature portföyü (15 dergi) | %8.9 | %9.4 | %3.4 |

Matematik istisnadır ve girişleri özetlerinden **daha az** değiştirilmiştir.
Tahmin yönteminin kendi hata payı **<3.5 puan** — her oran bu bantla okunmalı.

Bağımsız doğrulama (farklı yöntem, farklı korpus): Kobak vd. (2025) PubMed
özetleri için 2024'te **≥%13.5** tahmin ediyor; en iyi tek kelime ("potential")
oranı **5.2 puan** kaydırıyor.

Bölüm yönü yayımlanmış, sayıları değil: **özet, giriş, ilgili çalışmalar ve
sonuç** bölümleri yöntem-deney bölümlerinden daha çok değiştirilmiştir.

### İngilizce sözlük sinyalleri — ölçülmüş, çevrimdışı

Türkçenin yayımlanmış karşılığı olmayan iki sözlük sinyali İngilizce'de var:

- **Aşırı kelime seti** (Kobak vd., >15M PubMed özeti): `delve`, `underscore`,
  `intricate`, `meticulous`, `pivotal`, `comprehensive`, `crucial`, `insights` …
- **Ters sinyal, daha faydalı olan yarısı:** Thelwall & Kousha (1.25M tam metin)
  LLM'lerin **`thus` ve `moreover`** kelimelerinden *kaçındığını* ölçtü. İkisi de
  insan metninde yüksek sıklıkta bağlaç, dolayısıyla **eksikliği** pozitif
  kanıttır — model gerektirmez.

### Dil, lenf ve L2

İngilizce taramada üç ölçülmüş uyarı rapora eklenir:

- **Tür ve uzunluk:** Turnitin insanlıkta **0.86**, fen bilimlerinde **0.51**
  (Hadra vd. 2026, p=.0149). Uzunluk: 300–330 kelimede **0.87**, 450–550'de
  **0.56**. Tek belge puanı yanlış olur.
- **L1/L2:** Doğrulanmış vakaların **%82'si** İngilizcenin resmî dil olmadığı
  ülkelerden. Ana dilini İngilizce olmayan yazarın metni **insan yazımı olduğu
  halde** L1 ölçüte göre leksik olarak fakir görünür: CEFR A2 seviyesinde MTLD
  **58 ± 22**, L1 akademik medyan **94**.
- **Tekrar çalışması:** Liang vd. (2023) L2 yanlış pozitif oranını **%61.4**
  olarak raporladı; EACL 2026 aynı veri setinde **%23.1** buldu ve Çek
  non-native metinlerinde entropinin **yüksek** olduğunu gördü — **yön bile
  değişti**. Bu yüzden araç sabit bir yanlış pozitif oranı varsayamaz.

---

## Tez profili

`--profile thesis` (veya `similarity`/`compliance` komutları) şunları açıkça
kapatır:

* **Bölüm rolleri.** Başlıktan bölüm türü çıkarılır (Giriş / Yöntem / Bulgular /
  Tartışma / Sonuç / Kaynakça / Teşekkür / Ekler…). Kaynakça, teşekkür ve ön bölüm
  **AI ölçümüne ve benzerlik hesabına girmez**; özet bölümü yüksek, veri bölümü
  düşük önselle alır — çünkü özet en son ve en kalıplaşmış bölümdür.
* **Kurumsal benzerlik filtreleri.** `kaynakça hariç` · `alıntılar dahil` ·
  `5 kelimeden az eşleşmeler hariç` — İstanbul Üniversitesi kural dosyasındaki
  filtre setinin aynısı.
* **Kurum eşikleri.** YTÜ 15/20/2, İstanbul ve Başkent %20, ESÜ %30/15,
  Çukurova ve Akdeniz %30 — kaynak URL'leriyle birlikte raporlanır.
* **Uydurma eşik yok.** Yüzde, ölçülmüş Türkçe tez dağılımıyla karşılaştırılır:
  ortalama **%28.7** ± 11.58 (n=600; Toprak 2014).

---

## Sinyaller ve ağırlıkları

| Sinyal | Ağırlık | Gerekçe (sayı) |
|---|---|---|
| Likelihood-ratio (iki model) | 0.34 | Düzenlenmiş insan akademik yazımında FPR **%0.0–0.2** (LRR/Binoculars); token-istatistik yöntemleri **%99.8–100** (arXiv:2608.26710) |
| Sınıflandırıcı | 0.22 | Türkçede in-domain %0.95–0.97, out-of-domain **%0.35–0.42**; işe yaramıyorsa otomatik düşürülür |
| Stilometri | 0.20 | Model gerektirmez, dile özgü kalıplar |
| Ritim | 0.10 | Uzun metinde cümle uzunluğu düzgünlüğü |
| Ham perplexity | **0.08** | Resmî metinde FPR **%61–80** (EPO patent iddiaları, arXiv:2607.13044) |

12 kelimenin altındaki cümleler AI ölçümüne girmez: 10 token'da AUROC 0.62,
yanlış pozitiflerin mod uzunluğu 34 kelime (arXiv:2509.18880, arXiv:2603.23146).

### Neden bu ağırlıklar?

Varsayılanlar uydurulmadı, ölçüldü. Ham perplexity birincil sinyal iken
**%61–100 FPR** veren yöntemlerin taşıdığı bölüme düşürüldü ve yerine, o
literatürde sıfıra yakın FPR verdiği ölçülen aile kondu. Ayrıntı ve kaynaklar:
[LITERATURE.md §1.1](LITERATURE.md).

### Metin bütünlüğü

Homoglyph, sıfır genişlik karakter, yumuşak tire ve tekrarlı noktalama taranır.
Bulgu varsa AI sinyalleri güvenilmez işaretlenir: RAID'de homoglyph saldırısı
ortalama doğruluğu **%40.6** düşürüyor ve bu araç için neden tespit edilemeyeceğini
bilmek, neden tespit edildiğini iddia etmekten daha dürüst.

---

## İntihal: eşleşme sayısı değil, atıf durumu

2025–26 ölçek-büyüklüğünde bir çalışmada eşleşen makale çiftlerinin yalnızca
**%6.4**'ünde atıf bağlantısı bulundu (insan doğrulaması %7.0, CI 3.5–10.6) ve
eşleşen çiftler 4 katına çıkarken atıflı çiftler sabit kaldı (arXiv:2609.32963).
**Yani eşleşme sayısı suç sayısı değildir.** Bu yüzden her eşleşme Turnitin'in
dört grubundan birine yerleştirilir:

| Durum | Anlamı | Skor çarpanı |
|---|---|---|
| `cited_and_quoted` | Kaynak verilmiş, tırnak içinde — meşru | 0.10 |
| `missing_quotation` | Kaynak verilmiş ama tırnak içinde değil | 0.55 |
| `missing_citation` | Tırnak içinde ama kaynak verilmemiş | 0.75 |
| `not_cited_or_quoted` | Ne kaynak ne tırnak — ağır olan | 1.00 |

Kaynakça bölümü sayımdan çıkarılır; kendi tezine karşı tarama yapılmaz.

---

## Çıktı nasıl okunur

```
╭─ tez.md ───────────────────────────────────────────────╮
│ risk MEDIUM · 0.56   AI 0.41   intihal 0.54            │
│ 20 cümle · 147 kelime · 6 işaretli · 4 yüksek riskli   │
│ ort. perplexity 60.7 · burstiness ppl 50.9             │
│ modeller: ytu-ce-cosmos/turkish-gpt2, … | …           │
│ AI ölçümü yapılmayan cümle: 17                         │
╰───────────────────────────────────────────────────────╯
```

Benzerlik paneli:

```
- Alıntılar dahil: 20.41%   (Turnitin'in ana rakamı)
- Alıntılar hariç: 14.97%   (YTÜ eşiği %15)
- En yüksek tek kaynak: 13.61%  (kaynak2.md)
- Örtüşen 30 kelime / 147 · 5 eşleşme, 3 vaka (granülarite 1.667)
| Atıf durumu | Eşleşme |
| not_cited_or_quoted | 2 |
| cited_and_quoted | 3 |
```

`--annotate` aynı bilgiyi metnin içine koyar, orijinal düzeni bozmadan:

```
Ayrıca, model çıktısının burstiness değeri insan yazımına göre daha düzenlidir.⟦#3 yüksek | AI %38 | intihal %83 | oran 0.28 | sebep: discourse_marker,match_exact⟧
```

---

## MCP sunucusu

Tüm özellikler Model Context Protocol üzerinden bir asistana açılır — asistan
tezin nerede riskli olduğunu **ve neden** sorabilir.

```bash
checker mcp                        # stdio (Claude Desktop, Claude Code, …)
checker mcp --transport http       # streamable HTTP, port 8090
make mcp-test                      # gerçek istemciyle uçtan uca test
```

Claude Desktop `claude_desktop_config.json`:

```json
{ "mcpServers": {
    "checker": { "command": "/abs/path/checker/.venv/bin/checker", "args": ["mcp"] } } }
```

### Araçlar

| Araç | Ne yapar | Model gerekir |
|---|---|---|
| `checker_doctor` | ortamı, modelleri, hangi sinyalin düşeceğini bildirir | hayır |
| `checker_segment` | konumlu cümle bölütleme (model yok, anında) | hayır |
| `checker_similarity` | kurumsal filtrelerle benzerlik yüzdeleri + eşikler | hayır |
| `checker_patchwork` | yamalama biçimi: kaç karodan derlenmiş, şans tabanının üstünde mi | hayır |
| `checker_references` | kaynakça yapısal denetimi: hangi kayıtlar elle aranmalı | hayır |
| `checker_compliance` | YÖK kontrol listesi: beyan, atıfsız eşleşme, riskli bölüm | isteğe bağlı |
| `checker_matches` | atıf durumuyla eşleşmeler (`cited_and_quoted` meşru) | hayır |
| `checker_compare_revisions` | iki sürümün cümle bazlı farkı | hayır |
| `checker_scan` | tam analiz: cümle bazlı AI riski + intihal | evet |
| `checker_sentences` | paged cümle listesi, filtreli | isteğe bağlı |
| `checker_calibrate` | kendi korpusta işletim noktası ölçümü | evet |

`checker://disclaimer` kaynağı, aracın neleri iddia **edemeyeceğini** okur.

Dört tasarım kuralı:

- **Bulgu her zaman konumlu gelir.** Cümle indeksi, karakter/satır/sayfa ofseti
  ve düz dilde gerekçe birlikte seyahat eder; "bu cümle riskli" tek başına
  işe yaramaz.
- **Yükseklik sınırlı.** Bir tez raporu binlerce cümledir; her araç özet +
  en kötü N kaydı döndürür, gerisi için `checker_sentences` / `checker_matches`
  vardır.
- **Çökme saklanmaz, raporlanır.** Bir sinyal çalışamazsa (eksik model, Türkçe
  metne İngilizce sınıflandırıcı) `degraded_signals` içinde gerekçesiyle döner.
- **Bütün araçlar `read_only_hint`.** Hiçbiri diske yazmaz, istemci
  otomatik onaylayabilir.
- **Kural ihlali edilmez.** Araç "derece" (ne kadar) iddiasında bulunmaz ve
  %20'nin altında belge hükmü üretmez; bu iki kural araç talimatlarına ve
  `checker://disclaimer` kaynağına gömülüdür.

---

## Model yönetimi

| Rol | Türkçe | İngilizce |
|---|---|---|
| perplexity (gözlemci) | `ytu-ce-cosmos/turkish-gpt2` | `gpt2` |
| likelihood-ratio (icra) | `hakanbogan/gpt2-turkish-cased` | `gpt2-large` (yayımlanmış çift) |
| sınıflandıricı | `Hello-SimpleAI/chatgpt-detector-roberta` (İngilizce eğitilmiş → Türkçede otomatik düşürülür) | aynı |

Türkçe'de aynı tokenizer'a sahip bir model çifti yoktur; bu yüzden **gözlemcinin
token dizisi iki modele de verilir** (`score_token_ids`). Aksi halde "oran" iki
tokenizer'ın farkını ölçtü: TR AUC 0.375 → düzeltmeyle **0.944**.

Ölçülen işletim noktaları: TR 0.33 (insan medyanı 0.359, AI 0.257, FPR %0),
EN 0.901 (yayımlanmış eşik; FPR %0). Bunlar varsayılandır ama **sizin
koleksiyonunuz için `checker calibrate` ile yeniden ölçülmelidir**.

---

## Mimari

```
checker/
├── checker_app/
│   ├── domain/          saf katman: I/O ve model yok
│   │   ├── enums.py         RiskLevel, Signal, CitationStatus, MatchKind
│   │   ├── sections.py      tez bölüm rolleri (TR/EN başlıklar)
│   │   ├── models.py        Location, Sentence, SentenceReport, DocumentReport
│   │   ├── text.py          katlama, tokenizasyon, dil tespiti
│   │   └── segmentation.py  konumlu cümle bölütleme
│   ├── services/
│   │   ├── stylometry.py    model gerektirmeyen dilbilgisi sinyalleri
│   │   ├── lexicon.py       TR/EN kalıp sözlüğü
│   │   ├── perplexity.py    causal LM + burstiness (bağlamlı)
│   │   ├── ratio.py         iki model likelihood-ratio (Binoculars ailesi)
│   │   ├── detector.py      AI/Human sınıflandırıcı + kullanışlılık koruması
│   │   ├── integrity.py     homoglyph / sıfır genişlik karakter teşhisi
│   │   ├── plagiarism.py    shingle, şablon, near-match
│   │   ├── attribution.py   atıf durumu (Turnitin'in 4 grubu)
│   │   ├── similarity.py    kurumsal filtrelerle benzerlik yüzdeleri
│   │   ├── code_ast.py      Python AST + LaTeX normalizasyonu
│   │   ├── patchwork.py     yamalama biçimi (HyPlag GIT) + şans tabanı
│   │   ├── discourse.py     RST ilişki bağlaç profili
│   │   ├── crosslingual.py  çeviri dayanıklı çapa kümeleme
│   │   ├── reliability.py   ECE / Brier / güvenilirlik / sıcaklık
│   │   ├── references.py    kaynakça yapısal denetimi
│   │   ├── english_baselines.py  İngilizce ölçülmüş referanslar
│   │   ├── calibration.py   yerel işletim noktası ölçümü
│   │   ├── compliance.py    YÖK/kurumsal kontrol listesi
│   │   ├── scoring.py       sinyal füzyonu, açıklamalar, bölüm toplamları
│   │   ├── sources.py       dosya/dizin/URL/stdin, önbellek
│   │   ├── report.py        JSON / Markdown / işaretli metin
│   │   ├── runner.py        orkestrasyon ve bozuk sinyal yönetimi
│   │   └── model_fetch.py   yerel önbellek, .bin → safetensors
│   ├── mcp/
│   │   └── server.py        11 araç + disclaimer kaynağı (stdio / HTTP)
│   ├── config.py        tüm ayarlar (env ile)
│   └── cli.py
├── scripts/           bootstrap · calibrate · measure_ratio · test_mcp_stdio
├── tests/             399 test, model indirmeden
├── LITERATURE.md      sayı → karar eşlemesi
└── requirements.txt
```

Tasarım kararları:

- **Sinyaller bağımsız ve isteğe bağlı.** Biri çalışmazsa rapor üretilir, sinyal
  `degraded_signals` içinde gerekçesiyle listelenir, ağırlıklar yeniden dağıtılır
  ve `confidence` düşer.
- **Konum sözleşmesi.** Her cümle/eşleşme/bulgu karakter ofseti taşır.
- **Model arayüzü.** `LMBackend` protokolü sayesinde pencereleme ve iki-model
  mantığı 500 MB model indirmeden test edilir.
- **Kaynakça kendi kendine kaynak olmaz.** `--ref-dir .` verildiğinde belgenin
  kendisi ve bu aracın çıktıları otomatik elenir.

### Uygulama notları

- **Cümle bölütleme** `3,14`, `1.000,50`, `Dr.`, `vb.`, `J. Smith`, `U.S.`, URL'ler
  için bölmez; tırnak içi bölünmez; başlık/liste/kod/tablo/denklem yapısı korunur.
- **`.bin` kontrol noktaları.** `transformers` 4.5x, torch 2.6'dan eski
  sürümlerde yalnız `pytorch_model.bin` olan modelleri reddeder; `model_fetch.py`
  ağırlıkları `weights_only=True` ile safetensors'a çevirir.
- **Tokenizer/model uyuşmazlığı.** `hakanbogan/gpt2-turkish-cased` 50258 sözlüklü
  ama gömme matrisi 50257 geniş; pad ve hedef token kimlikleri kırpılır, aksi
  halde `gather` CUDA'da cihaz hatası verir.
- **Kendi bağlamına koşulma tuzağı.** Filtrelenmiş cümle listelerinde bağlam,
  belge indeksiyle değil **liste konumuyla** belirlenir; aksi halde her cümlenin
  perplexity'si ~1 çıkar (regresyon testiyle kilitli).

---

## Sınırlar

- Tek cümle kanıt değildir; bulgu, insanın gözden geçirmesi için sıralamadır.
- **Türkçe'de dedektörler ölçülmüş biçimde başarısız.** Altıntop (2026,
  DOI 10.56493/nkusbmyo.1866431) 8 dedektörü, hiçbir aşamada AI kullanılmadan
  yazılmış **5.715 kelimelik** Türkçe bir akademik metin üzerinde denedi:

  | Dedektör | %100 insan metne verdiği karar |
  |---|---|
  | Justdone | **%89 AI** |
  | ZeroGPT | ~%80 AI |
  | Sidekicker | "yapay zekâ üretimi işaretleri" |
  | MyDetector | "büyük olasılıkla %40 AI" |
  | TruthScan | %40 AI |
  | QuillBot / Smodin / Copyleaks | %0 AI |

  Aynı metin intihal motorlarında %9 (yalnız alıntı filtresi) ve %4 (alıntı +
  kaynakça hariç) çıktı — yani sorun metinde değil, dedektörde.

  Daha keskin olan dil etkisi: Derrida'nın *Plato's Pharmacy*'nin **aynı**
  sayfası ZeroGPT'de 1972 Fransızca orijinalde **%0 AI**, 2012 Türkçe
  çevirisinde **%73.25 AI**. Bir Türkçe aracı İngilizce performanstan kalibre
  edilemez.

  Yayımlanmış Türkçe dedektör sayıları **vardır** (AUROC %99.31, özgüllük
  %94.16 → **FPR %5.84**, DOI 10.28948/ngumuh.1930411) ama haber/özet/ödev
  kayıtlarında ölçülmüştür ve teze en yakın iki alanda en düşük sonuçlar
  oradadır: **akademik %95.21**, **mevzuat-hukuk %94.99**. Bu yüzden bu araç
  başka bir çalışmadan eşik almaz; `checker calibrate` ile kendi korpusunuzda
  ölçer.
- Ham perplexity kendi başına kullanılmaz; rapor onu betimleyici bir istatistik
  olarak gösterir.
- Paraphrase tamamen yeniden yazılmış metni bu araç bulmaz (shingle recall'ı
  paraphrase'ta 0.35→0.08 düşüyor — PAN 2025).
- İntihal yalnız verilen kaynaklara karşı çalışır; genel web taraması yoktur.
- Eşikler varsayılan olarak **tutucudur**: daha çok kaçırır, daha az suçlar.
- YÖK'ün tezlerde AI kullanımına dair oran belirleyen bir mevzuatı 2026 itibarıyla
  yoktur; YÖK'ün Mayıs 2024 Etik Rehber'i beyan yükümlülüğü getirir, yüzde getirmez.
- **Jüri de bu işte zayıf.** 63 öğretim üyesi Alman tez parçalarında %57 (AI
  metni) / %64 (insan metni) doğru tanıdı, profesyonel düzeydeki AI metninde
  %20'nin altında kaldı; insan ile makine dedektör arasında anlamlı fark yok
  (DOI 10.1016/j.iree.2025.100321). Tez alanı diğer alanlardan da zor
  (DeBERTa F1: tez 85.88, hikâye 90.05, deneme 60.66).
- **Kaynakça denetimi yapısal olduğu için var olmayı kanıtlayamaz.** DOI
  çözmez, CrossRef sorgulamaz. Yalnız "elle aranacak kayıt listesi" üretir.
  Tezler için ölçülmüş uydurma kaynak oranı **yayımlanmamıştır**.
- **Puanlar olasılık değil.** 0–1 ölçekte olmaları bir olasılık anlamına gelmez;
  ECE/Brier ancak etiketli bir korpusla ölçülür ve 30 örnek altında hesaplanmaz.
- **Kısa belge nötr kanıt değildir.** Kelime sayısının tek başına ROC-AUC'i
  0.68 olarak ölçülmüştür.
- **Leksikal zenginlik tek yönlü AI işareti değildir.** Hafif AI düzenlemesi
  insan değerinin üstüne çıkarır, üretim altına indirir.
- **Kurumlar dedektörü kapatıyor.** Vanderbilt (2023), UCLA ve UC San Diego
  (2024), Waterloo (Eylül 2025), Curtin (Ocak 2026) Turnitin AI dedektörünü
  devre dışı bıraktı. İngiltere Ombudsmanı CS072502'de şikâyeti "haklı"
  buldu. Bu araç bir yedek değil, jüriye eline verilecek **kanıt tabanıdır**.

---

## Geliştirme

```bash
make check          # lint + typecheck + test
make fmt
make mcp-test       # gerçek MCP istemcisiyle uçtan uca (11 araç)
.venv/bin/python scripts/measure_ratio.py --lang tr   # sinyal ayrımını ölç
.venv/bin/python scripts/calibrate.py                 # perplexity kalibrasyonu
bash scripts/seed_literature.sh                       # makaleleri MCP korpusuna al
```