#!/usr/bin/env bash
# checker-app kararlarını besleyen literatürü paper MCP korpusuna alır.
#
# Makaleler "checker-literature" projesine eklenir. Ingestion yavaştır (MinerU,
# CPU'da sayfa başına saniyeler), bu yüzden sırayla ve arka planda çalışır.
#
#   nohup bash scripts/seed_literature.sh > /tmp/seed.log 2>&1 &
#   tail -f /tmp/seed.log
#
# MCP sunucusu yeniden adlandırma sonrası yeniden başlatılana kadar proje
# doğrudan paper_app katmanı üzerinden yazılır (aynı tablolar, aynı veri).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$(cd "$HERE/../paper" && pwd)"
PY="$PAPER/.venv/bin/python"
PROJECT="checker-literature"

# id :: neden bu makale
IDS=(
  "2405.07940::RAID benchmark: esik kalibrasyonu ve FPR olcumleri (ana referans)"
  "2608.26710::Stil karistirici: 135k duzenlenmis insan metninde FP olcumu"
  "2510.06805::PAN 2025 intihal gorevi: n-gram geri cagrilimi, granularity"
  "2609.32963::Olcek buyuklugunde metin yeniden kullanimi: %6.4 atif baglantisi"
  "2509.18880::DivEye: kisa metin AUROC dususu, CEFR seviyesine gore FP"
  "2607.13044::The Perplexity Trap: resmi metinlerde %61-80 FPR"
  "2608.11256::Akademik butunluk icin AI tespiti neden basarisiz"
  "2510.25057::Same Same But Different: kod AST normalizasyonu dayanikligi"
  "2604.25778::Kod intihali: AST AUROC 0.893 vs n-gram 0.765"
  "2511.00416::PADBen/RADAR: cumle cifti vs tekil cumle, laundering"
  "2602.08934::StealthRL: uyarlanabilir paraphrase saldirisi"
  "2605.06098::AST deseni vs LLM ile kod klon tespiti"
)

cd "$PAPER" || exit 1

for entry in "${IDS[@]}"; do
  id="${entry%%::*}"
  why="${entry##*::}"
  echo "=== [$(date +%H:%M:%S)] ingest $id — $why"
  if "$PY" -m paper_app.cli ingest "$id" >> /tmp/opencode/ck/seed_ingest.log 2>&1; then
    echo "    ok: $id"
  else
    echo "    BASARISIZ: $id (log: /tmp/opencode/ck/seed_ingest.log)"
    continue
  fi
done

echo "=== proje baglantisi"
"$PY" - <<'PY'
import asyncio

from paper_app.db.project_repository import ProjectRepository
from paper_app.db.session import session_scope

PROJECT = "checker-literature"
#: id :: not  (projeye eklenirken saklanacak okuma notu)
NOTES = {
    "2405.07940": "RAID: naive esik (tau=0.5) ile FPR %23-100; homoglyph altinda ortalama dogruluk dususu %40.6. Esik sabit degil, FPR'a gore kalibre edilmeli.",
    "2608.26710": "Token-istatistik dedektorleri duzenlenmis insan akademik yaziminda %99.8-100 FP; Binoculars %0.0, LRR %0.2. Tez icin bu ayrim belirleyici.",
    "2510.06805": "Leksikal n-gram taban paragrafada 0.50, paraphrase kumesinde 0.23; precision korunur, recall 0.35->0.08. Granularity cezasi: F1/log2(1+gran).",
    "2609.32963": "Eslesen ciftlerin yalnizca %6.4'u atif baglantili (insan dogrulamasi %7.0). Kaynakca/boilerplate sayimdan once temizlenmeli.",
    "2509.18880": "10 token'da AUROC 0.62, 256 token'da 0.88. Kisa cumle tespit edilemez; FP medyani 34 kelime.",
    "2607.13044": "EPO patent iddialarinda FPR: DetectGPT %80.5, Binoculars %78.3, Fast-DetectGPT %61.3. Resmi dil en kotu durum.",
    "2608.11256": "Dürüst AI düzeltmesi, kaçınmaya göre çok daha sık işaretleniyor (Pangram %80.1, GPTZero %48.5).",
    "2510.25057": "Ölü kod kaldırma + ifade düzeni normalizasyonu ekleme saldırılarını çözer (%80-99 ayrım). AI yeniden yazımı hepsini kırar (delta<=0.09).",
    "2604.25778": "ConPlag2: syntax AUROC 0.893 vs n-gram 0.765; yorum/boşluk/import temizliği ucuz ve yüksek getirili.",
    "2511.00416": "Cümle çifti girişi tekil cümleye göre +0.1-0.2 AUC; paraphrase sonrası kaynak atıfı AUC 0.49-0.53 (çözülemez).",
    "2602.08934": "StealthRL: ortalama AUROC 0.789->0.432; Binoculars AUROC 0.055. Ayarlanabilir saldırı altında token-kimlik yöntemleri çöker.",
    "2605.06098": "AST-desen eşleme LLM tanımadan üstün (F1 0.74 vs 0.35).",
}


async def main() -> None:
    async with session_scope() as session:
        repo = ProjectRepository(session)
        project = await repo.require(PROJECT)
        for paper_id, note in NOTES.items():
            try:
                # Id'yi once cozumle: link_paper_ids unresolved donerse hicbir
                # sey yazilmamis gibi gorunur.
                resolved, _missing = await repo.resolve_paper_ids([paper_id])
                if not resolved:
                    print(f"  ! {paper_id}: korpusta yok (ingest basarisiz olmus olabilir)")
                    continue
                added = await repo.add_papers(
                    project, resolved, added_by="checker", note=note
                )
                print(f"  + {resolved[0]}: eklendi" if added else f"  = {resolved[0]}: zaten var")
            except Exception as exc:  # noqa: BLE001 - raporlanacak
                print(f"  ! {paper_id}: {type(exc).__name__}: {exc}")
        info = await repo.list_projects()
        for item in info:
            if item.slug == PROJECT:
                print(f"  proje: {item.name} -> {item.paper_count} makale")


asyncio.run(main())
PY
echo "=== [$(date +%H:%M:%S)] bitti"