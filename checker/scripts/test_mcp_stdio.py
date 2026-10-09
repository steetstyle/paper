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


#: An English thesis, to check that the language gate selects the English
#: baseline and the English institutional bands. Clean English dissertations
#: measure 9% +/- 6% against 28.7% for Turkish, so a mix-up shows in the payload.
ENGLISH_THESIS = """# Abstract

This study investigates how undergraduate students revise their writing when
automated feedback is made available during drafting. The findings suggest that
iterative feedback helps students revise more effectively, although the present
study does not establish the underlying mechanism.

# 1. INTRODUCTION

Prior work has largely treated writing as a single-shot activity in which
students compose text and then submit it for evaluation. We show that this
approach understates the revision process substantially.

# 2. METHOD

Participants were recruited from two introductory writing courses and were
randomly assigned to one of three conditions. Each session lasted ninety minutes
and was recorded with written consent obtained beforehand.

# 3. RESULTS

The revision scores in the iterative-feedback condition were higher than those in
the control condition, though the difference was modest and should be read with
care.

# 4. DISCUSSION

The findings suggest that iterative feedback helps students revise more
effectively, but the present study does not establish the mechanism involved.

# REFERENCES

Smith, J. (2019). Automated writing evaluation. Journal of Writing Research.
"""


#: A separate English reference. It cannot be the thesis itself: the loader drops
#: a document from its own reference set, which would leave nothing to compare.
ENGLISH_SOURCE = """# Reference

Writing has often been treated as a single activity that ends when the text is
submitted for evaluation, and this assumption has shaped instruction and
assessment for decades.

The revision scores in the iterative-feedback condition were higher than those
in the control condition, though the difference was modest and should be read
with care.
"""


#: An English source whose numbers and proper nouns survive translation, and the
#: Turkish translation of it. The word matcher sees nothing between them; the
#: anchor pass is the only thing that can.
ENGLISH_CITED = """# Methods (English)

Participants were recruited from two introductory writing courses and 412
students completed the full protocol. Cohen (1988) reports an inter-rater
agreement of 0.80 as the minimum acceptable value. Data collection ran between
2019 and 2021 and produced 23 usable response sets after 7 exclusions. The
instrument was scored using the rubric published by Kaufman and Smith (2019).
"""

TURKISH_TRANSLATION = """# YÖNTEM

Katılımcılar iki temel yazma dersinden ve 412 öğrenciden toplandı ve
protokolün tamamını tamamladı. Cohen (1988), kabul edilebilir en düşük
değer olarak 0.80 aralıklar arası uyum raporlamaktadır. Veri toplama 2019 ile
2021 arasında yürütülmüş ve 7 çıkarım sonrası 23 kullanılabilir yanıt seti
üretilmiştir. Ölçek, Kaufman ve Smith (2019) tarafından yayımlanan rubrik
kullanılarak puanlanmıştır.
"""


def fixtures() -> tuple[Path, Path, Path, Path]:
    """Sample files, or the ones the user named. Sync: writing files inside the
    event loop would block the client transport."""
    if len(sys.argv) < 2:
        workdir = Path("/tmp/checker-mcp-demo")
        workdir.mkdir(parents=True, exist_ok=True)
        thesis = workdir / "tez.md"
        source = workdir / "kaynak.md"
        english = workdir / "thesis-en.md"
        english_source = workdir / "ref-en.md"
        thesis.write_text(THESIS, encoding="utf-8")
        source.write_text(SOURCE, encoding="utf-8")
        english.write_text(ENGLISH_THESIS, encoding="utf-8")
        english_source.write_text(ENGLISH_SOURCE, encoding="utf-8")
        (workdir / "cited-en.md").write_text(ENGLISH_CITED, encoding="utf-8")
        (workdir / "tr-translation.md").write_text(TURKISH_TRANSLATION, encoding="utf-8")
        print(f"örnek tez yazıldı: {thesis} (+ İngilizce örnek)")
        return thesis, source, english, english_source
    thesis = Path(sys.argv[1]).resolve()
    source = (
        Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else thesis.parent / "kaynak.md"
    )
    english = thesis.parent / "thesis-en.md"
    english_source = thesis.parent / "ref-en.md"
    if not english.exists():
        english.write_text(ENGLISH_THESIS, encoding="utf-8")
    if not english_source.exists():
        english_source.write_text(ENGLISH_SOURCE, encoding="utf-8")
        (workdir / "cited-en.md").write_text(ENGLISH_CITED, encoding="utf-8")
        (workdir / "tr-translation.md").write_text(TURKISH_TRANSLATION, encoding="utf-8")
    return thesis, source, english, english_source


async def main() -> int:
    thesis, source, english, english_source = fixtures()

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

        # English mode: the language gate must switch both the baseline and the
        # institutional bands. A mix-up would apply Turkish thresholds to an
        # English thesis, where the measured clean corpus is 3x lower.
        english_scan = await call(
            "checker_scan",
            {
                "path": str(english),
                "profile": "thesis",
                "use_ratio": False,
                "use_perplexity": False,
                "use_classifier": False,
                "min_level": "none",
                "top": 1,
            },
        )
        context = english_scan.get("verdict", {}).get("english_context")
        if not context:
            failures.append("İngilizce belgede english_context yok")
        else:
            presence = context["ai_presence"]
            print(
                f"İngilizce: dil={english_scan.get('document', {}).get('language')} "
                f"AI={presence['arxiv_cs_abstracts']} (CS) / "
                f"{presence['arxiv_mathematics_abstracts']} (mat) · "
                f"hata payı={presence['estimator_error_percentage_points']} puan"
            )
            for key, expected in (
                ("arxiv_cs_abstracts", 0.225),
                ("arxiv_mathematics_abstracts", 0.077),
            ):
                if presence[key] != expected:
                    failures.append(f"İngilizce AI yaygınlığı yanlış: {key}")
        print(
            f"  tür/uzunluk: {'0.86' in (context or {}).get('genre_and_length_note', '')} · "
            f"L2 tekrar: {'%82' in (context or {}).get('l1_l2_note', '')} · "
            f"tekrar çalışması: {'23.1' in (context or {}).get('replication_note', '')}"
        )

        english_similarity = await call(
            "checker_similarity", {"path": str(english), "refs": [str(english_source)]}
        )
        similarity_block = english_similarity.get("similarity", {})
        if similarity_block.get("applied_baseline") != "english":
            failures.append(
                f"İngilizce belge Türkçe tabanı kullandı: "
                f"{similarity_block.get('applied_baseline')}"
            )
        baseline = similarity_block.get("english_baseline") or {}
        if baseline.get("mean_percent") != 9.0:
            failures.append("İngilizce taban 9% olmalı")
        if baseline.get("optimal_cutoff_percent") != 15.0:
            failures.append("İngilizce ölçülmüş optimal eşik 15% olmalı")
        band_names = " ".join(b["institution"] for b in similarity_block.get("institutional_bands", []))
        print(
            f"İngilizce benzerlik: taban=%{baseline.get('mean_percent')} "
            f"optimal_eşik={baseline.get('optimal_cutoff_percent')} "
            f"kurumlar={len(similarity_block.get('institutional_bands', []))}"
        )
        if "Virginia Tech" not in band_names:
            failures.append("İngilizce kurum bantları uygulanmadı")
        if "YTÜ" in band_names:
            failures.append("Türkçe kurum bantları İngilizce belgeye sızdı")

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

        # The blind spot this closes: a translated passage reports 0% overlap
        # under word matching, and the anchor pass is what finds it.
        cited_en = thesis.parent / "cited-en.md"
        tr_translation = thesis.parent / "tr-translation.md"
        translation = await call(
            "checker_similarity", {"path": str(tr_translation), "refs": [str(cited_en)]}
        )
        block = translation.get("similarity", {})
        lexical = block.get("incl_quotes_percent")
        cross = block.get("cross_lingual") or {}
        print(
            f"çeviri: kelime eşleşmesi={lexical}% · "
            f"çap kümeleri={cross.get('clusters_found')} · "
            f"güven={cross.get('confidence_counts')}"
        )
        if not cross.get("clusters_found"):
            failures.append("çevrilmiş bir aktarım çapa geçişiyle bulunamadı")
        counts = cross.get("confidence_counts", {})
        if counts.get("review", 0) == cross.get("clusters_found"):
            failures.append("çeviri bulgusu yalnız 'review' seviyesinde kalmamalı")
        # The ceiling has to travel with it, or the zero reads as exoneration.
        if "%74.10" not in (cross.get("evidence") or ""):
            failures.append("çapraz dil tavanı (%74.10) raporla birlikte gitmiyor")
        if "ALT SINIRDIR" not in (cross.get("caveat") or ""):
            failures.append("çapraz dil alt sınır uyarısı eksik")

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