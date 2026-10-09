"""Live end-to-end check: a real MCP client over stdio.

Not a unit test - it launches ``checker mcp`` as a subprocess and speaks the
protocol to it, which is the only way to catch a server that imports fine but
cannot actually serve. Run it with:

    .venv/bin/python scripts/test_mcp_stdio.py /path/to/thesis.md
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

VENV_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python")
THESIS = """# Özet

Bu çalışmada, öğrencilerin akademik yazım performansının yapay zeka araçlarıyla
desteklenmesi üzerine kapsamlı bir inceleme sunulmaktadır. Ayrıca, bu tür
araçların kullanımının öğrenme süreçleri üzerindeki etkisi detaylı bir analiz
ile ele alınmıştır.

# 1. GİRİŞ

Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. Model
eğitimi üç ayrı koşul altında titizlikle yürütülmüştür ve sonuçlar kaydedilmiştir.

# Kaynakça

Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri. Eğitim Bilimleri Dergisi.

Zhang, W. (2023). Attention mechanisms revisited. Neural Networks. doi:10.9999

Doe, A. (2099). Future perspectives on tutoring. Journal of Applied Research.
"""
SOURCE = """# Kaynak

Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. Model
eğitimi üç ayrı koşul altında titizlikle yürütülmüştür ve sonuçlar kaydedilmiştir.
"""


def fixtures() -> tuple[Path, Path]:
    """Sample files, or the ones the user named. Sync: writing files inside the
    event loop would block the client transport."""
    if len(sys.argv) < 2:
        workdir = Path("/tmp/checker-mcp-demo")
        workdir.mkdir(parents=True, exist_ok=True)
        thesis = workdir / "tez.md"
        source = workdir / "kaynak.md"
        thesis.write_text(THESIS, encoding="utf-8")
        source.write_text(SOURCE, encoding="utf-8")
        print(f"örnek tez yazıldı: {thesis}")
        return thesis, source
    thesis = Path(sys.argv[1]).resolve()
    source = (
        Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else thesis.parent / "kaynak.md"
    )
    return thesis, source


async def main() -> int:
    thesis, source = fixtures()

    params = StdioServerParameters(
        command=VENV_PYTHON,
        args=["-m", "checker_app.mcp.server"],
        env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"},
    )

    failures: list[str] = []

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        init = await session.initialize()
        print(f"bağlandı: {init.server_info.name} {init.server_info.version}")
        if init.instructions:
            print(f"talimatlar: {len(init.instructions)} karakter")

        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        print(f"araçlar ({len(names)}): {', '.join(names)}")
        expected = {
            "checker_segment", "checker_scan", "checker_similarity",
            "checker_compliance", "checker_sentences", "checker_matches",
            "checker_compare_revisions", "checker_calibrate", "checker_doctor",
            "checker_patchwork", "checker_references",
        }
        if set(names) != expected:
            failures.append(f"araç kümesi beklenenden farklı: {set(names) ^ expected}")

        resources = await session.list_resources()
        print(f"kaynaklar: {[str(r.uri) for r in resources.resources]}")
        read_result = await session.read_resource("checker://disclaimer")
        # Client side carries ``.text``; the in-process server side uses
        # ``.content``. Both are read here so a shape change is visible.
        text = "".join(
            (getattr(c, "text", None) or getattr(c, "content", "") or "")
            for c in read_result.contents
        )
        print(f"disclaimer okundu: {len(text)} karakter, RAID geçiyor: {'RAID' in text}")
        if "RAID" not in text:
            failures.append("disclaimer kaynağı RAID'e atıf yapmıyor")

        async def call(name: str, args: dict) -> dict:
            result = await session.call_tool(name, args)
            if getattr(result, "isError", False):
                failures.append(f"{name} hata döndü: {result.content}")
                return {}
            return result.structured_content or {}

        doctor = await call("checker_doctor", {"language": "tr"})
        print(
            f"doctor: dil={doctor.get('language')} "
            f"classifier_kullanilabilir={doctor.get('models', {}).get('classifier_usable')}"
        )
        if doctor.get("models", {}).get("classifier_usable") is not False:
            failures.append("doctor Türkçe için sınıflandırıcıyı kullanılamaz demeli")

        segment = await call("checker_segment", {"path": str(thesis), "max_items": 5})
        print(
            f"segment: {segment.get('sentences')} cümle, dil={segment.get('language')}, "
            f"ilk={segment.get('items', [{}])[0].get('text', '')[:40]!r}"
        )
        if not segment.get("sentences"):
            failures.append("segment cümle döndürmedi")

        similarity = await call(
            "checker_similarity", {"path": str(thesis), "refs": [str(source)]}
        )
        stats = similarity.get("similarity", {}).get("stats", {})
        print(
            f"similarity: alıntılar dahil="
            f"{similarity.get('similarity', {}).get('incl_quotes_percent')}% "
            f"eşleşme={stats.get('match_count')} vaka={stats.get('cases')}"
        )
        if not stats:
            failures.append("similarity istatistikleri boş")

        compliance = await call(
            "checker_compliance", {"path": str(thesis), "refs": [str(source)]}
        )
        items = {i["code"]: i for i in compliance.get("compliance", {}).get("items", [])}
        print(
            f"compliance: beyan={items.get('ai_disclosure', {}).get('status')} "
            f"gerekli={compliance.get('compliance', {}).get('blocking_items')}"
        )
        if items.get("ai_disclosure", {}).get("status") != "required":
            failures.append("beyan eksikliği 'required' olarak işaretlenmeli")

        matches = await call(
            "checker_matches",
            {"path": str(thesis), "refs": [str(source)], "unattributed_only": True},
        )
        print(
            f"matches: {matches.get('total_matching')} atıfsız eşleşme, "
            f"durumlar={matches.get('attribution_counts')}"
        )

        scan = await call(
            "checker_scan",
            {
                "path": str(thesis),
                "refs": [str(source)],
                "use_ratio": False,
                "use_perplexity": False,
                "use_classifier": False,
                "min_level": "none",
                "top": 3,
            },
        )
        print(
            f"scan: risk={scan.get('verdict', {}).get('risk')} "
            f"bulgu={len(scan.get('findings', []))} "
            f"bölüm={len(scan.get('sections', []))}"
        )
        if not scan.get("findings"):
            failures.append("scan bulgu döndürmedi")
        verdict = scan.get("verdict", {})
        print(
            f"  AI kapsamı %{verdict.get('ai_share', 0) * 100:.1f} · "
            f"taban altı={verdict.get('below_ai_reporting_floor')} · "
            f"belge iddiası={verdict.get('document_ai_claim')}"
        )
        if verdict.get("below_ai_reporting_floor") and verdict.get("document_ai_claim") is not None:
            failures.append("taban altındaken belge AI iddiası üretilmemeliydi")
        if not verdict.get("below_ai_reporting_floor") and not verdict.get(
            "document_ai_claim"
        ):
            failures.append("tabandan yüksek payda belge AI iddiası üretilmeliydi")

        patchwork = await call(
            "checker_patchwork", {"path": str(thesis), "refs": [str(source)]}
        )
        pw = patchwork.get("patchwork", {})
        print(
            f"patchwork: skor={pw.get('score')} karo={pw.get('tile_count')} "
            f"taban={pw.get('chance_baseline')} biçim={str(pw.get('shape'))[:40]}"
        )
        if pw.get("chance_baseline") != 0.15:
            failures.append("HyPlag ölçülmüş şans tabanı 0.15 olmalı")
        if not pw.get("caveats"):
            failures.append("yamalama raporu kendi sınırlarını taşımıyor")

        audit = await call("checker_references", {"path": str(thesis), "show_all": True})
        refs = audit.get("references", {})
        if "error" in audit:
            print(f"references: {audit['error']}")
        else:
            print(
                f"references: {refs.get('references_scanned')} kayıt · "
                f"kimliksiz={refs.get('without_identifier')} · "
                f"gözden geçirilecek={refs.get('needs_review')} · "
                f"kanıt={len(refs.get('evidence', []))} kaynak"
            )
            if not refs.get("caveats"):
                failures.append("kaynakça denetimi kendi sınırlarını taşımıyor")

        context = verdict.get("style_context", {})
        print(
            f"bağlam: uzunluk={context.get('length_confound', {}).get('regime')} "
            f"kelime_sayisi_auc="
            f"{context.get('length_confound', {}).get('measured', {}).get('word_count_only_auroc')} "
            f"· leksikal={'var' if context.get('lexical_richness') else 'yok (200 token eşiği)'} "
            f"· bağlaç={context.get('discourse', {}).get('connectives')}"
        )
        if context.get("length_confound", {}).get("measured", {}).get(
            "word_count_only_auroc"
        ) != 0.68:
            failures.append("kelime sayısı karıştırıcısı bağlamda değil")
        if not context.get("discourse", {}).get("measured_direction"):
            failures.append("söylem yönü ölçümü taşınmıyor")

        # The measured Turkish false-positive rates are the core evidence for
        # this tool's design, so a lost caveat is a failed run.
        turkish = verdict.get("caveats", {}).get("turkish_fpr", "")
        print(
            f"TR FPR kanıtı: %89={'%89' in turkish} "
            f"dil_salınımı={'%73.25' in turkish} "
            f"yayımlanmış_FPR={'%5.84' in turkish} "
            f"akademik={'%95.21' in turkish}"
        )
        for needle, label in (
            ("%89 AI", "Justdone'ın %89'u"),
            ("%73.25", "diller arası salınım"),
            ("%5.84", "yayımlanmış Türkçe FPR"),
            ("%95.21", "akademik alan sonucu"),
        ):
            if needle not in turkish:
                failures.append(f"TR FPR kanıtı eksik: {label}")

    print()
    if failures:
        print("BAŞARISIZ:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("TÜM MCP TESTLERİ GEÇTİ")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))