"""``checker`` command line interface.

checker scan tez.md --ref-dir ./kaynaklar
checker scan makale.pdf --ref https://arxiv.org/abs/1706.03762 --json rapor.json
checker compare v1.md v2.md                     # iki sürüm arası cümle farkı
checker segment tez.md                          # yalnız bölütleme (model indirmez)
checker doctor                                  # bağımlılık ve model durumu
checker models                                  # kullanılacak modeller
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from checker_app.config import CheckerSettings, get_settings, reload_settings
from checker_app.domain.enums import RiskLevel
from checker_app.domain.models import DocumentReport
from checker_app.logging import bind_run_id, configure_logging, get_logger
from checker_app.services.calibration import calibrate as calibrate_corpus
from checker_app.services.calibration import save_calibration
from checker_app.services.compliance import build_compliance_report
from checker_app.services.report import (
    compliance_markdown,
    diff_markdown,
    similarity_markdown,
    to_annotated_text,
    to_json,
    to_markdown,
)
from checker_app.services.runner import ScanRequest, ScanRunner
from checker_app.services.scoring import AI_REPORTING_FLOOR
from checker_app.services.sources import SourceLoader

app = typer.Typer(
    name="checker",
    help="Metni cümle cümle böler; AI yazım riski ve intihal eşleşmelerini konumlarıyla raporlar.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = get_logger("checker.cli")

_LEVEL_CHOICES = ["none", "low", "medium", "high", "critical"]


def _settings(verbose: bool, json_logs: bool) -> CheckerSettings:
    settings = reload_settings()
    configure_logging("DEBUG" if verbose else settings.log_level, json_logs)
    bind_run_id()
    return settings


def _level(value: str) -> RiskLevel:
    try:
        return RiskLevel(value)
    except ValueError as exc:
        raise typer.BadParameter(f"geçersiz seviye: {value}. Seçenekler: {_LEVEL_CHOICES}") from exc


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Ayrıntılı log.")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="JSON log satırları.")] = False,
) -> None:
    """Genel seçenekler."""
    ctx.obj = {"verbose": verbose, "json_logs": json_logs}


@app.command()
def scan(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Analiz edilecek dosya; '-' ise stdin okunur.")],
    ref: Annotated[
        list[str] | None,
        typer.Option("-r", "--ref", help="Kaynak dosya/URL/metin. Birden fazla kez verilebilir."),
    ] = None,
    ref_dir: Annotated[
        list[str] | None,
        typer.Option("--ref-dir", help="Kaynak dizini (tüm metin dosyaları)."),
    ] = None,
    ref_glob: Annotated[str, typer.Option("--ref-glob", help="Dizin filtresi.")] = "*",
    ref_exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Kaynak listesinden çıkarılacak desen (tekrarlanabilir)."),
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="JSON raporu bu dosyaya yazılır.")
    ] = None,
    md_out: Annotated[
        Path | None, typer.Option("--md", help="Markdown raporu bu dosyaya yazılır.")
    ] = None,
    annotate_out: Annotated[
        Path | None,
        typer.Option("--annotate", help="Cümle sonuna risk etiketi eklenmiş metni yazar."),
    ] = None,
    compliance_out: Annotated[
        Path | None,
        typer.Option("--compliance", help="YÖK/kurumsal uyum kontrol listesini yazar."),
    ] = None,
    lang: Annotated[str, typer.Option("--lang", help="tr | en | auto")] = "auto",
    no_perplexity: Annotated[
        bool, typer.Option("--no-perplexity", help="Perplexity kapalı.")
    ] = False,
    no_classifier: Annotated[
        bool, typer.Option("--no-classifier", help="Sınıflandırıcı kapalı.")
    ] = False,
    no_plagiarism: Annotated[bool, typer.Option("--no-plagiarism", help="İntihal kapalı.")] = False,
    no_code: Annotated[
        bool, typer.Option("--no-code", help="Kod/denklem karşılaştırması kapalı.")
    ] = False,
    no_ratio: Annotated[
        bool,
        typer.Option(
            "--no-ratio",
            help="Likelihood-ratio (iki model) sinyalini kapatır: daha az RAM, daha zayıf ayrım.",
        ),
    ] = False,
    profile: Annotated[
        str,
        typer.Option(
            "--profile", help="document | thesis (tez: bölüm rolleri + kurumsal filtreler)."
        ),
    ] = "document",
    ppl_model: Annotated[str | None, typer.Option("--ppl-model", help="Perplexity modeli.")] = None,
    ratio_model: Annotated[
        str | None,
        typer.Option("--ratio-model", help="Likelihood-ratio'un ikinci (performer) modeli."),
    ] = None,
    classifier_model: Annotated[
        str | None, typer.Option("--classifier-model", help="Sınıflandırıcı modeli.")
    ] = None,
    min_level: Annotated[
        str, typer.Option("--min-level", help="Raporda gösterilecek en düşük risk.")
    ] = "low",
    top: Annotated[int, typer.Option("--top", help="En fazla kaç cümle gösterilsin.")] = 25,
    show_all: Annotated[bool, typer.Option("--all", help="Tüm cümleleri göster.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Yalnız dosyalara yaz.")] = False,
    fail_on: Annotated[
        str | None, typer.Option("--fail-on", help="Bu risk seviyesinde çıkış kodu 1.")
    ] = None,
) -> None:
    """Bir metni cümle cümle inceler ve risk raporu üretir."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    if profile not in ("document", "thesis"):
        raise typer.BadParameter("--profile: 'document' veya 'thesis' olmalı")
    settings.profile = cast(Literal["document", "thesis"], profile)
    level = _level(min_level)

    loader = SourceLoader(settings)
    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)

    references = []
    for spec in ref or []:
        references.append(loader.load_reference(spec))
    for directory in ref_dir or []:
        references.extend(loader.load_directory(directory, ref_glob))
    if not references and not no_plagiarism:
        console.print(
            "[yellow]Kaynak verilmedi:[/yellow] intihal analizi yapılmayacak "
            "(--ref, --ref-dir veya --ref ile URL ekleyin)."
        )

    runner = ScanRunner(settings, sources=loader.to_sources(references))
    report = runner.scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            language=None if lang == "auto" else lang,
            use_perplexity=not no_perplexity,
            use_classifier=not no_classifier,
            use_plagiarism=not no_plagiarism,
            use_code=not no_code,
            use_ratio=not no_ratio,
            perplexity_model=ppl_model,
            ratio_model=ratio_model,
            classifier_model=classifier_model,
        )
    )

    if json_out:
        json_out.write_text(to_json(report), encoding="utf-8")
    if md_out:
        md_out.write_text(
            to_markdown(report, min_level=level, include_all=show_all), encoding="utf-8"
        )
    if compliance_out:
        compliance_out.write_text(
            compliance_markdown(report, build_compliance_report(report, raw_text=loaded.text)),
            encoding="utf-8",
        )
    if annotate_out:
        annotate_out.write_text(to_annotated_text(report, loaded.text), encoding="utf-8")
    if not quiet:
        _render_report(report, level=level, limit=top, show_all=show_all)

    if fail_on and report.risk.at_least(_level(fail_on)):
        raise typer.Exit(code=1)


@app.command()
def compare(
    ctx: typer.Context,
    left: Annotated[str, typer.Argument(help="Önceki sürüm.")],
    right: Annotated[str, typer.Argument(help="Yeni sürüm.")],
    md_out: Annotated[Path | None, typer.Option("--md", help="Markdown çıktısı.")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
) -> None:
    """İki sürümü cümle düzeyinde karşılaştırır (kendi kendine intihal için de kullanılır)."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    loader = SourceLoader(settings)
    runner = ScanRunner(settings)

    left_text = sys.stdin.read() if left == "-" else loader.load_document(left).text
    right_text = loader.load_document(right).text
    diffs = runner.diff(left_text, right_text)

    markdown = diff_markdown(diffs)
    if md_out:
        md_out.write_text(markdown, encoding="utf-8")
    payload = [
        {
            "op": d.op.value,
            "similarity": round(d.similarity, 4),
            "left_index": d.left_index,
            "right_index": d.right_index,
            "left_text": d.left_text,
            "right_text": d.right_text,
            "left_location": d.left_location.label() if d.left_location else None,
            "right_location": d.right_location.label() if d.right_location else None,
        }
        for d in diffs
    ]
    if json_out:
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title="Cümle düzeyi fark", show_lines=False)
    table.add_column("", width=1)
    table.add_column("Konum", no_wrap=True)
    table.add_column("Cümle")
    colors = {"+": "green", "-": "red", "~": "yellow", "=": "dim"}
    marks = {"insert": "+", "delete": "-", "replace": "~", "equal": "="}
    for diff in diffs:
        if diff.op.value == "equal":
            continue
        location = diff.left_location or diff.right_location
        mark = marks[diff.op.value]
        table.add_row(
            Text(mark, style=colors[mark]),
            location.label() if location else "",
            (diff.left_text or diff.right_text or "")[:120],
        )
    console.print(table)
    counts: dict[str, int] = {}
    for diff in diffs:
        counts[diff.op.value] = counts.get(diff.op.value, 0) + 1
    console.print(
        f"eşit {counts.get('equal', 0)} · yeni {counts.get('insert', 0)} · "
        f"silinen {counts.get('delete', 0)} · değişen {counts.get('replace', 0)}"
    )


@app.command()
def similarity(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Analiz edilecek dosya.")],
    ref: Annotated[list[str] | None, typer.Option("-r", "--ref", help="Kaynak dosya/URL.")] = None,
    ref_dir: Annotated[list[str] | None, typer.Option("--ref-dir", help="Kaynak dizini.")] = None,
    ref_glob: Annotated[str, typer.Option("--ref-glob", help="Dizin filtresi.")] = "*",
    ref_exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Kaynak listesinden çıkarılacak desen (tekrarlanabilir)."),
    ] = None,
    excl_quotes: Annotated[
        bool, typer.Option("--excl-quotes", help="Alıntıları sayımdan çıkarır (YTÜ filtresi).")
    ] = False,
    keep_references: Annotated[
        bool, typer.Option("--keep-references", help="Kaynakçayı sayıma dahil eder.")
    ] = False,
    min_match: Annotated[
        int, typer.Option("--min-match-words", help="Kurumsal filtre: eşik kelime sayısı.")
    ] = 5,
    lang: Annotated[str, typer.Option("--lang", help="tr | en | auto")] = "auto",
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
    md_out: Annotated[Path | None, typer.Option("--md", help="Markdown çıktısı.")] = None,
) -> None:
    """Turnitin tarzı benzerlik yüzdesi, kurumsal filtreler ve eşiklerle."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    settings.plagiarism.include_quotes = not excl_quotes
    settings.plagiarism.exclude_references = not keep_references
    settings.plagiarism.institutional_min_match_words = max(1, min_match)
    settings.profile = "thesis"

    loader = SourceLoader(settings)
    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)
    references = _load_references(loader, ref, ref_dir, ref_glob, exclude=loaded.path,
                 exclude_globs=tuple(ref_exclude or ()))
    if not references:
        raise typer.BadParameter("kaynak gerekli: --ref veya --ref-dir")

    runner = ScanRunner(settings, sources=loader.to_sources(references))
    report = runner.scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            language=None if lang == "auto" else lang,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
            use_code=False,
        )
    )
    markdown = similarity_markdown(report)
    if md_out:
        md_out.write_text(markdown, encoding="utf-8")
    if json_out:
        json_out.write_text(to_json(report), encoding="utf-8")

    console.print(Panel(markdown.replace("# ", "").split("\n", 1)[1][:1200]))
    if report.similarity is not None:
        table = Table(title="Kaynak dağılımı")
        table.add_column("Kaynak")
        table.add_column("Kelime", justify="right")
        table.add_column("Pay %", justify="right")
        table.add_column("Eşleşme", justify="right")
        table.add_column("Atıfsız", justify="right")
        for share in report.similarity.per_source:
            table.add_row(
                share.name,
                str(share.matched_words),
                f"{share.share_percent:.2f}",
                str(share.match_count),
                str(share.uncited_matches),
            )
        console.print(table)


@app.command()
def patchwork(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Analiz edilecek dosya.")],
    ref: Annotated[list[str] | None, typer.Option("-r", "--ref", help="Kaynak dosya/URL.")] = None,
    ref_dir: Annotated[list[str] | None, typer.Option("--ref-dir", help="Kaynak dizini.")] = None,
    ref_exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Kaynak listesinden çıkarılacak desen (tekrarlanabilir)."),
    ] = None,
    lang: Annotated[str, typer.Option("--lang", help="tr | en | auto")] = "auto",
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
) -> None:
    """Yamalama (patchwriting) biçimi: metin kaç parçadan derlenmiş, şans tabanının üstünde mi.

    Yüzde "ne kadar" der, bu komut "nasıl derlenmiş" diye yanıtlar. HyPlag'ın Greedy
    Identifier Tiles mantığı (JCDL 2019, arXiv:1906.11761); şans tabanı 1.000.000 rastgele
    belge çifti üzerinde ölçülmüştür (GIT ≥ 0,15).
    """
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    settings.profile = "thesis"

    loader = SourceLoader(settings)
    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)
    references = _load_references(
        loader, ref, ref_dir, "*", exclude=loaded.path, exclude_globs=tuple(ref_exclude or ())
    )
    if not references:
        raise typer.BadParameter("kaynak gerekli: --ref veya --ref-dir")

    runner = ScanRunner(settings, sources=loader.to_sources(references))
    report = runner.scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            language=None if lang == "auto" else lang,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
            use_code=False,
        )
    )
    if report.patchwork is None:
        raise typer.BadParameter("yamalama raporu üretilemedi")
    payload = report.patchwork
    if json_out:
        json_out.write_text(
            json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    body = Text()
    body.append(f"skor {payload.score:.4f}\n", style="bold")
    body.append(
        f"şans tabanı {payload.chance_baseline} → "
        + ("tabanın üstünde\n" if payload.above_chance else "tabanın altında\n"),
        style="dim",
    )
    body.append(f"karo: {payload.tile_count} · kelime: {payload.tile_words}\n", style="dim")
    body.append(f"en büyük karo: {payload.largest_tile_words} kelime\n", style="dim")
    body.append(f"biçim: {payload.shape}\n\n")
    body.append(f"{payload.reading}\n\n", style="dim")
    for caveat in payload.caveats:
        body.append(f"· {caveat}\n", style="yellow")
    console.print(Panel(body, title="Yamalama (patchwriting) profili", expand=False))


@app.command()
def references(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Analiz edilecek tez dosyası.")],
    show_all: Annotated[bool, typer.Option("--all", help="Temiz kayıtları da gösterir.")] = False,
    limit: Annotated[int, typer.Option("--limit", help="Gösterilecek kayıt sayısı.")] = 40,
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
) -> None:
    """Kaynakça denetimi: hangi girdiler insan eliyle aranmalı, ve neden.

    Yalnızca yapısal denetimdir — DOI çözülmez, CrossRef sorgulanmaz. Uydurma kaynakça
    alanın en iyi kanıtlanmış sorunu: The Lancet 2026 taramasında 2,5M makalede 4.046
    uydurma kaynak, oran üç yılda 12 kat arttı (2023'te 1/2.828 → 2026'da 1/277).
    """
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    settings.profile = "thesis"

    loader = SourceLoader(settings)
    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)
    report = ScanRunner(settings).scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
            use_plagiarism=False,
            use_code=False,
        )
    )
    audit = report.references
    if audit is None or not audit.total:
        console.print("[yellow]Kaynakça bölümü bulunamadı ya da boş.[/yellow]")
        raise typer.Exit(code=0)

    summary = audit.summary()
    if json_out:
        json_out.write_text(
            json.dumps(audit.to_dict(limit=limit), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    console.print(
        Panel(
            f"{summary['references_scanned']} kayıt tarandı · "
            f"{summary['with_identifier']} kalıcı kimlikli · "
            f"{summary['needs_review']} elden geçmeli\n"
            f"ok {summary['risk_counts']['ok']} · "
            f"note {summary['risk_counts']['note']} · "
            f"review {summary['risk_counts']['review']}\n"
            f"{audit.reading()}",
            title="Kaynakça denetimi",
            expand=False,
        )
    )

    rows = audit.entries if show_all else audit.flagged
    table = Table(title=f"Kayıtlar ({len(rows)}/{audit.total})", show_lines=False)
    table.add_column("#", justify="right", width=4)
    table.add_column("Durum", width=7)
    table.add_column("Yıl", justify="right", width=5)
    table.add_column("Kimlik", no_wrap=True, overflow="fold")
    table.add_column("Kayıt", overflow="fold", max_width=64)
    table.add_column("İşaretler", overflow="fold")
    for entry in rows[:limit]:
        table.add_row(
            str(entry.index),
            entry.risk,
            str(entry.year or "-"),
            entry.identifier or "-",
            entry.text[:150],
            ", ".join(entry.flags) or "-",
        )
    console.print(table)
    console.print("[dim]Kanıt: The Lancet 2026 (4.046 uydurma kaynak / 2.810 makale); "
                  "arXiv:2601.18724; arXiv:2602.05867; DOI 10.1038/d41586-026-00969-z[/dim]")


@app.command()
def compliance(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Tez dosyası.")],
    ref: Annotated[list[str] | None, typer.Option("-r", "--ref", help="Kaynak dosya/URL.")] = None,
    ref_dir: Annotated[list[str] | None, typer.Option("--ref-dir", help="Kaynak dizini.")] = None,
    ref_exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Kaynak listesinden çıkarılacak desen (tekrarlanabilir)."),
    ] = None,
    lang: Annotated[str, typer.Option("--lang", help="tr | en | auto")] = "auto",
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
    md_out: Annotated[Path | None, typer.Option("--md", help="Markdown çıktısı.")] = None,
) -> None:
    """YÖK/kurumsal uyum kontrol listesi: beyan, atıf durumu, AI yoğun bölümler."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    settings.profile = "thesis"

    loader = SourceLoader(settings)
    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)
    references = _load_references(loader, ref, ref_dir, None, exclude=loaded.path,
                       exclude_globs=tuple(ref_exclude or ()))

    runner = ScanRunner(settings, sources=loader.to_sources(references))
    report = runner.scan(
        ScanRequest(
            path=loaded.path,
            text=loaded.text,
            language=None if lang == "auto" else lang,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
        )
    )
    result = build_compliance_report(report, raw_text=loaded.text)
    if json_out:
        json_out.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if md_out:
        md_out.write_text(compliance_markdown(report, result), encoding="utf-8")

    table = Table(title="Uyum kontrol listesi")
    table.add_column("Durum", width=12)
    table.add_column("Kural")
    table.add_column("Bulgu")
    marks = {
        "ok": "[green]uygun[/green]",
        "note": "[dim]bilgi[/dim]",
        "review": "[yellow]gözden geçir[/yellow]",
        "required": "[red]gerekli[/red]",
    }
    for item in result.items:
        table.add_row(marks.get(item.status, item.status), item.rule, item.evidence)
    console.print(table)
    console.print(
        f"gerekli madde: {result.blocking} · atıfsız eşleşme: {len(result.uncited_matches)}"
    )
    for note in result.notes:
        console.print(Text(note, style="dim italic"))


@app.command()
def calibrate(
    ctx: typer.Context,
    human_dir: Annotated[
        Path, typer.Option("--human-dir", help="Kendi insan yazımınızın bulunduğu dizin.")
    ],
    machine_dir: Annotated[
        Path | None,
        typer.Option("--machine-dir", help="Bilinen AI metinlerinin bulunduğu dizin (opsiyonel)."),
    ] = None,
    lang: Annotated[str | None, typer.Option("--lang", help="tr | en | auto")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Ölçüm sonucunu JSON olarak yaz.")
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Ölçüm JSON çıktısı.")] = None,
) -> None:
    """Kendi insan metniniz üzerinde işletim noktasını ölçer.

    RAID: hazır eşikler FPR %23-100 veriyor; tek savunulabilir nokta kendi
    kayıtlarınızda ölçülmüş olanıdır. Yön bile ölçülür: bazı model çiftlerinde
    oranın işareti terstir (ölçülen AUC 0.25).
    """
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    if not human_dir.is_dir():
        raise typer.BadParameter(f"dizin bulunamadı: {human_dir}")

    extensions = (".txt", ".md", ".markdown", ".rst")
    human_files = sorted(
        p for p in human_dir.rglob("*") if p.suffix.lower() in extensions
    )
    machine_files: list[Path] = []
    if machine_dir is not None:
        if not machine_dir.is_dir():
            raise typer.BadParameter(f"dizin bulunamadı: {machine_dir}")
        machine_files = sorted(
            p for p in machine_dir.rglob("*") if p.suffix.lower() in extensions
        )
    if not human_files:
        raise typer.BadParameter(f"metin dosyası yok: {human_dir}")

    console.print(
        f"{len(human_files)} insan + {len(machine_files)} AI dosyası ölçülüyor "
        f"(model yükleniyor, ilk çalıştırmada birkaç dakika sürebilir)…"
    )
    result = calibrate_corpus(
        settings,
        human_files=human_files,
        machine_files=machine_files,
        language=None if lang in (None, "auto") else lang,
    )

    table = Table(title="Ölçüm")
    table.add_column("Küme", width=16)
    table.add_column("Cümle", justify="right")
    table.add_column("Oran medyanı", justify="right")
    table.add_column("p10-p90", justify="right")
    if result.human:
        table.add_row("insan", str(result.human.sentences),
                      f"{result.human.median_ratio:.3f}",
                      f"{result.human.q10:.3f}-{result.human.q90:.3f}")
    if result.machine:
        table.add_row("AI-benzeri", str(result.machine.sentences),
                      f"{result.machine.median_ratio:.3f}",
                      f"{result.machine.q10:.3f}-{result.machine.q90:.3f}")
    console.print(table)
    if result.human:
        console.print(
            f"modeller: {result.human.observer} (gözlemci) / "
            f"{result.human.performer} (icra)"
        )
    if result.auc is not None:
        console.print(f"AUC(oran) = {result.auc:.3f} · yön: {result.direction}")
    if result.fpr_at_center is not None:
        console.print(
            f"verilen merkezde insan metni FPR = %{round(result.fpr_at_center * 100)}"
        )
    console.print(f"önerilen merkez: {result.to_dict()['recommended_center']}")
    if result.reliability is not None:
        reliability = result.reliability
        if reliability.measured:
            console.print(
                f"kalibrasyon: {reliability.interpretation}",
                style="yellow" if (reliability.ece or 0) > 0.10 else "dim",
            )
            table = Table(title="Güvenilirlik (reliability) kovaları")
            table.add_column("Aralık", width=12)
            table.add_column("n", justify="right", width=5)
            table.add_column("Ort. skor", justify="right", width=10)
            table.add_column("Gerçek oran", justify="right", width=11)
            table.add_column("Fark", justify="right", width=8)
            for row in reliability.bins:
                table.add_row(
                    f"{row.low:.1f}–{row.high:.1f}",
                    str(row.count),
                    f"{row.mean_score:.3f}",
                    f"{row.positive_rate:.3f}",
                    f"{row.mean_score - row.positive_rate:+.3f}",
                )
            console.print(table)
        else:
            console.print(f"kalibrasyon: {reliability.interpretation}", style="yellow")
        for caveat in result.reliability.caveats:
            console.print(Text(f"  · {caveat}", style="dim"))
    for warning in result.warnings:
        console.print(Text(f"! {warning}", style="yellow"))
    if out is not None:
        save_calibration(out, result)
        console.print(f"yazıldı: {out}")
    if json_out is not None:
        json_out.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _warn_if_unreadable(target: str, word_count: int) -> None:
    """Say "the file is unreadable" rather than "the thesis is empty".

    A file that is not a PDF at all - a failed download that saved an HTML error
    page under a .pdf name, a truncated transfer, a hand rename - used to raise
    pypdf's own error out of the reader constructor. With that guarded, the only
    remaining symptom is zero words, which reads as an empty document rather than
    a broken file. Measured on a real case: a file saved as ``.pdf`` that was
    entirely HTML.
    """
    if word_count:
        return
    path = Path(target)
    if path.suffix.lower() not in {".pdf"}:
        return
    console.print(
        "[yellow]Bu PDF'ten metin çıkarılamadı. Dosya bozuk ya da PDF değil "
        "(örneğin başarısız bir indirmeden kalan HTML). "
        "Sıfır cümle, boş tez demek değildir — belge hiç okunamadı.[/yellow]"
    )


@app.command()
def segment(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Bölütlenecek dosya.")],
    min_level: Annotated[
        str, typer.Option("--min-level", help="Gösterilecek en düşük risk.")
    ] = "none",
    json_out: Annotated[Path | None, typer.Option("--json", help="JSON çıktısı.")] = None,
) -> None:
    """Yalnızca cümle bölütlemesini gösterir (model indirmez, anında çalışır)."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    _level(min_level)  # validate early so the failure is not mid-scan
    loader = SourceLoader(settings)
    from checker_app.domain.segmentation import SentenceSplitter  # noqa: PLC0415

    loaded = loader.load_stdin() if target == "-" else loader.load_document(target)
    result = SentenceSplitter(settings.segmentation).split(loaded.text)

    table = Table(title=f"Bölütleme — {loaded.path}")
    table.add_column("#", justify="right", width=4)
    table.add_column("Tür", width=10)
    table.add_column("Konum", no_wrap=True)
    table.add_column("Cümle")
    for sentence in result.sentences:
        table.add_row(
            str(sentence.index),
            sentence.location.block_type.value,
            sentence.location.label(),
            sentence.snippet(140),
        )
    console.print(table)
    console.print(
        f"{len(result.sentences)} cümle · {result.word_count} kelime · "
        f"dil {result.language} ({result.language_confidence:.2f}) · "
        f"{len(result.blocks)} blok"
    )
    _warn_if_unreadable(target, result.word_count)
    if json_out:
        json_out.write_text(
            json.dumps(
                {
                    "language": result.language,
                    "language_confidence": round(result.language_confidence, 3),
                    "sentence_count": len(result.sentences),
                    "word_count": result.word_count,
                    "page_count": result.page_count,
                    "sentences": [
                        {
                            "index": s.index,
                            "text": s.text,
                            "words": s.word_count,
                            "location": {
                                "char_start": s.location.char_start,
                                "char_end": s.location.char_end,
                                "line_start": s.location.line_start,
                                "line_end": s.location.line_end,
                                "page": s.location.page,
                                "paragraph": s.location.paragraph_index,
                                "section": list(s.location.section_path),
                                "block_type": s.location.block_type.value,
                            },
                        }
                        for s in result.sentences
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


@app.command()
def models(ctx: typer.Context) -> None:
    """Kullanılacak modelleri ve yerel önbellek durumunu gösterir."""
    settings = get_settings()
    table = Table(title="Modeller")
    table.add_column("Sinyal")
    table.add_column("Model")
    table.add_column("Rol")
    table.add_row("perplexity", settings.ai.perplexity_model_tr, "Türkçe causal LM")
    table.add_row("perplexity", settings.ai.perplexity_model_en, "İngilizce causal LM")
    table.add_row("classifier", settings.ai.classifier_model, "Human/AI sınıflandırıcı")
    console.print(table)

    cache = Path(settings.ai.cache_dir) if settings.ai.cache_dir else _hf_cache_dir()
    if cache.exists():
        for model_id in (
            settings.ai.perplexity_model_tr,
            settings.ai.perplexity_model_en,
            settings.ai.classifier_model,
        ):
            folder = cache / ("models--" + model_id.replace("/", "--"))
            state = "yerel" if folder.exists() else "indirilecek"
            console.print(f"  {model_id}: {state}")
    else:
        console.print(f"[dim]HuggingFace önbelleği bulunamadı: {cache}[/dim]")


@app.command()
def mcp(
    ctx: typer.Context,
    transport: Annotated[
        str, typer.Option("--transport", help="stdio | http")
    ] = "stdio",
    host: Annotated[str, typer.Option("--host", help="HTTP dinlenecek adres.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="HTTP portu.")] = 8090,
) -> None:
    """MCP sunucusunu başlatır (Claude Desktop / Claude Code / herhangi bir MCP istemcisi).

    Araçlar: checker_doctor, checker_segment, checker_scan, checker_similarity,
    checker_compliance, checker_sentences, checker_matches,
    checker_compare_revisions, checker_calibrate.
    """
    try:
        from checker_app.mcp.server import run as run_mcp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise typer.BadParameter(
            f"MCP sunucusu için `mcp` paketi gerekli: {exc}\n"
            "pip install -r requirements.txt"
        ) from exc
    if transport not in ("stdio", "http"):
        raise typer.BadParameter("--transport: 'stdio' veya 'http' olmalı")
    _ = ctx
    run_mcp(transport=transport, host=host, port=port)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Bağımlılıkları, model erişimini ve ayarları denetler."""
    options = ctx.obj or {"verbose": False, "json_logs": False}
    settings = _settings(bool(options["verbose"]), bool(options["json_logs"]))
    rows: list[tuple[str, bool, str]] = []

    rows.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    for module, purpose, extra in (
        ("torch", "perplexity", "ai"),
        ("transformers", "perplexity + sınıflandırıcı", "ai"),
        ("bs4", "HTML kaynakları", "html"),
        ("pypdf", "PDF kaynakları", "pdf"),
        ("httpx", "URL kaynakları", None),
    ):
        try:
            __import__(module)
            ok = True
            version = getattr(sys.modules[module], "__version__", "kurulu")
        except ImportError:
            ok = False
            version = f"yok — pip install 'checker-app[{extra}]'" if extra else "yok"
        rows.append((module, ok, f"{purpose}: {version}"))

    console.print(Panel("checker doctor", expand=False))
    table = Table(show_header=True)
    table.add_column("Bileşen")
    table.add_column("Durum")
    table.add_column("Not")
    for name, ok, note in rows:
        table.add_row(name, "[green]ok[/green]" if ok else "[red]eksik[/red]", note)
    console.print(table)

    console.print(
        f"\nperplexity: `{settings.language_model('tr')}` (tr) · "
        f"`{settings.language_model('en')}` (en)"
    )
    console.print(f"classifier: `{settings.ai.classifier_model}`")
    console.print(
        "Model ilk kullanımda HuggingFace üzerinden indirilir; "
        "indirme çevrimdışıysa perplexity/sınıflandırıcı devre dışı kalır ve raporda "
        '"çalışmayan sinyaller" olarak görünür.'
    )


# ------------------------------------------------------------------- rendering
def _load_references(
    loader: SourceLoader,
    ref: list[str] | None,
    ref_dir: list[str] | None,
    ref_glob: str | None,
    *,
    exclude: str = "",
    exclude_globs: Sequence[str] = (),
) -> list:
    """Load references, dropping the document itself.

    ``--ref-dir .`` is the natural thing to type, and it includes the file being
    checked. Comparing a thesis with itself yields a 100% similarity that looks
    like a finding and is not one.
    """
    loaded = [loader.load_reference(spec) for spec in (ref or [])]
    for directory in ref_dir or []:
        loaded.extend(loader.load_directory(directory, ref_glob or "*"))
    if not exclude:
        return loaded
    target = Path(exclude).resolve()
    kept = []
    for item in loaded:
        try:
            same = item.kind == "file" and Path(item.path).resolve() == target
        except OSError:  # pragma: no cover - defensive
            same = False
        if same:
            console.print(f"[dim]kaynak atlandı (belgenin kendisi): {item.path}[/dim]")
            continue
        kept.append(item)
    return kept


def _hf_cache_dir() -> Path:
    """Where huggingface_hub keeps snapshots (``HF_HOME`` wins if set)."""
    import os  # noqa: PLC0415

    root = os.environ.get("HF_HOME")
    return Path(root) / "hub" if root else Path.home() / ".cache" / "huggingface" / "hub"


def _render_context(report: DocumentReport, verdict: Text) -> None:
    """Document-level context, appended to the verdict panel.

    These three are reported and deliberately *not* scored, and each line says
    why in one clause: a reader who sees a number needs to know whether it is a
    verdict, a description, or a warning.
    """
    if report.discourse is not None:
        verdict.append(f"bağlam · {report.discourse.reading()}\n", style="dim")

    stats = report.style_stats
    if stats is None:
        return
    if stats.lexical_richness is not None:
        _, label = stats.lexical_richness.closest_role()
        verdict.append(
            f"bağlam · leksikal zenginlik TTR {stats.lexical_richness.type_token_ratio:.3f} / "
            f"hapax {stats.lexical_richness.hapax_ratio:.3f} / "
            f"yoğunluk {stats.lexical_richness.lexical_density:.3f} "
            f"(ölçülen en yakın rol: {label}) — tek yönlü AI işareti değildir: cila "
            "insan değerinin üstüne çıkar, üretim altına indirir\n",
            style="dim",
        )
    if stats.length_confound is not None:
        verdict.append(
            f"bağlam · uzunluk {stats.length_confound.word_count} kelime "
            f"[{stats.length_confound.regime}] — kelime sayısının tek başına "
            "ROC-AUC 0.68 olduğu ölçülmüştür\n",
            style="dim",
        )
    if report.english_context is not None:
        style = report.english_context.style
        verdict.append(
            f"İngilizce referans · AI yaygınlığı arXiv CS %22.5 / matematik %7.7 "
            f"(hata payı <3.5 puan) · insan MTLD {style['document_level']['mtld_mean']:.0f} "
            f"(L2 A2 seviyesi {style['l2_reference']['a2_mtld']:.0f})\n",
            style="dim",
        )


def _render_report(report: DocumentReport, *, level: RiskLevel, limit: int, show_all: bool) -> None:
    color = report.risk.color
    verdict = Text()
    verdict.append(f"{report.path}\n", style="bold")
    verdict.append(
        f"risk {report.risk.value.upper()} · {report.risk_score:.2f}   ",
        style=color,
    )
    verdict.append(
        f"AI {report.ai_score:.2f}"
        + (" (betimleyici)" if report.below_ai_reporting_floor else "")
        + "   "
    )
    verdict.append(f"intihal {report.plagiarism_score:.2f}\n")
    verdict.append(
        f"{report.sentence_count} cümle · {report.word_count} kelime · "
        f"{report.flagged_sentences} işaretli · {report.high_risk_sentences} yüksek riskli · "
        f"{report.matched_sentences} eşleşen cümle\n",
        style="dim",
    )
    if report.mean_perplexity is not None:
        verdict.append(
            f"ort. perplexity {report.mean_perplexity:.1f} · "
            f"burstiness ppl {report.perplexity_burstiness} · uzunluk {report.length_burstiness}\n",
            style="dim",
        )
    # The reporting floor is the reason the AI number above can be zero while
    # sentences are still listed, so it is stated rather than left to be inferred.
    verdict.append(
        f"AI kapsamı %{report.ai_share * 100:.1f} ({report.ai_flagged_words} kelime) "
        + (
            f"— raporlama tabanının (%{AI_REPORTING_FLOOR * 100:.0f}) altında, belge düzeyinde "
            "AI hükmü verilmiyor\n"
            if report.below_ai_reporting_floor
            else "— tabanın üstünde\n"
        ),
        style="dim",
    )
    if report.models:
        verdict.append("modeller: " + ", ".join(report.models.values()) + "\n", style="dim")
    if report.turkish_fpr and report.language == "tr":
        # Only worth a line in the terminal when the document is Turkish: these
        # are the rates measured on Turkish academic prose specifically.
        verdict.append("kanıt · Türkçe ölçülmüş dedektör FPR'ları raporun sonunda\n", style="dim")
    _render_context(report, verdict)
    if report.degraded_signals:
        verdict.append("çalışmayan: " + "; ".join(report.degraded_signals) + "\n", style="yellow")
    console.print(Panel(verdict, border_style=color, expand=False))

    sections = Table(title="Bölümler", expand=False)
    sections.add_column("Bölüm")
    sections.add_column("Cümle", justify="right")
    sections.add_column("AI", justify="right")
    sections.add_column("İntihal", justify="right")
    sections.add_column("Yüksek", justify="right")
    for section in report.sections:
        sections.add_row(
            section.title,
            str(len(section.sentence_indices)),
            f"{section.ai_score:.2f}",
            f"{section.plagiarism_score:.2f}",
            str(section.high_risk_count),
        )
    if report.sections:
        console.print(sections)

    rows = [s for s in report.sentences if show_all or s.risk.at_least(level)]
    rows.sort(key=lambda s: (-s.risk_score, s.index))
    rows = rows[:limit] if limit else rows

    table = Table(title=f"Cümleler ({len(rows)}/{report.sentence_count})")
    table.add_column("#", justify="right", width=4)
    table.add_column("Konum", no_wrap=True, overflow="fold")
    table.add_column("Cümle", overflow="fold", max_width=70)
    table.add_column("Risk", width=9)
    table.add_column("AI", justify="right", width=4)
    table.add_column("Oran", justify="right", width=5)
    table.add_column("PPL", justify="right", width=5)
    table.add_column("İnt", justify="right", width=4)
    table.add_column("Nedenler")

    for sentence in rows:
        reasons = [f.message for f in sentence.findings[:2]] or ["-"]
        table.add_row(
            str(sentence.index),
            sentence.location.label(),
            sentence.snippet(110),
            Text(sentence.risk.value, style=sentence.risk.color),
            f"{sentence.ai_percent}",
            f"{sentence.ratio.score:.2f}" if sentence.ratio else "-",
            f"{sentence.perplexity.perplexity:.0f}" if sentence.perplexity else "-",
            f"{sentence.plagiarism_percent}"
            if sentence.plagiarism and sentence.plagiarism.has_match
            else "-",
            "\n".join(reasons),
        )
    console.print(table)
    console.print(Text(report.disclaimer, style="dim italic"))
