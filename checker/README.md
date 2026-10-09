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

## Söylem profili

Rapor, dört ilişki sınıfını (atıf / sıralama / açımlama / neden) TR/EN
bağlaçlarıyla sayar ve 100 kelimeye normalize eder. Yön ölçülmüştür: insan
yazarlar atıf ve sıralamayı, LLM'ler açımlamayı fazla kullanır (RACE,
arXiv:2604.04932). Önemli olan, bu yapının **düzenlemeden sonra da ayakta
kalması**: insan→AI cilalama kosinüs benzerliği 0.92 ± 0.08.

Tek başına zayıf (ölçülen motif F1'si 0.49–0.61), bu yüzden **hüküm üretmez**,
yalnız bağlam verir.

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
│   │   ├── reliability.py   ECE / Brier / güvenilirlik / sıcaklık
│   │   ├── references.py    kaynakça yapısal denetimi
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
├── tests/             250 test, model indirmeden
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
- **Türkçe'de hiçbir dedektörün yayımlanmış AUROC/FPR değeri yok.** Buradaki
  sayılar bu aracın kendi küçük örneklemleriyle ölçülmüştür.
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