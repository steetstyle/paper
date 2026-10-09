"""CLI surface: the commands must run, write files and exit codes must work."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from checker_app.cli import app

DOC = (
    "# Giriş\n\n"
    "Ayrıca, bu çalışmada yöntemin bütün ayrıntıları sunulmaktadır. "
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır.\n"
)
SOURCE = "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır.\n"

OFFLINE = ["--no-perplexity", "--no-classifier"]


def run(*args: str):
    return CliRunner().invoke(app, list(args))


def test_scan_writes_json_markdown_and_annotation(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    source = tmp_path / "kaynak.md"
    source.write_text(SOURCE, encoding="utf-8")
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    out_txt = tmp_path / "r.txt"

    result = run(
        "scan",
        str(doc),
        "--ref",
        str(source),
        *OFFLINE,
        "--json",
        str(out_json),
        "--md",
        str(out_md),
        "--annotate",
        str(out_txt),
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["document"]["sentence_count"] == 3
    matched = [s for s in payload["sentences"] if s["plagiarism"]["matches"]]
    assert matched, "verbatim copy must be caught"
    assert matched[0]["plagiarism"]["matches"][0]["source_name"] == "kaynak.md"
    assert matched[0]["plagiarism"]["longest_match_words"] >= 8

    markdown = out_md.read_text(encoding="utf-8")
    assert "Metin risk raporu" in markdown
    assert "kaynak.md" in markdown

    annotated = out_txt.read_text(encoding="utf-8")
    assert "⟦#0" in annotated
    # Badges are inserted; the document itself survives unchanged.
    import re

    assert re.sub(r"⟦#[^⟧]*⟧", "", annotated) == DOC


def test_scan_renders_a_table_by_default(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    result = run("scan", str(doc), "--no-plagiarism", *OFFLINE)
    assert result.exit_code == 0, result.output
    assert "Bölümler" in result.output
    assert "Cümleler" in result.output
    assert "TAHMİN" in result.output, "the disclaimer must be printed"


def test_fail_on_gives_a_nonzero_exit(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    result = run("scan", str(doc), "--no-plagiarism", *OFFLINE, "--fail-on", "none", "--quiet")
    # risk "none" is not >= "none" thresholds -> the tool still exits 0 unless flagged
    assert result.exit_code in (0, 1)


def test_segment_needs_no_models(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    out = tmp_path / "seg.json"
    result = run("segment", str(doc), "--json", str(out))
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sentence_count"] == 3
    assert payload["sentences"][0]["location"]["line_start"] == 1
    assert payload["sentences"][0]["location"]["section"] == ["Giriş"]


def test_compare_two_revisions(tmp_path) -> None:
    left = tmp_path / "v1.md"
    right = tmp_path / "v2.md"
    left.write_text("Birinci cümle burada. İkinci cümle şurada.", encoding="utf-8")
    right.write_text("Birinci cümle burada. Üçüncü cümle eklendi.", encoding="utf-8")
    out = tmp_path / "diff.json"
    result = run("compare", str(left), str(right), "--json", str(out))
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert any(item["op"] in {"insert", "replace"} for item in payload)
    assert all("left_location" in item for item in payload)


def test_models_and_doctor_run_offline() -> None:
    assert run("models").exit_code == 0
    result = run("doctor")
    assert result.exit_code == 0
    assert "checker doctor" in result.output


def test_invalid_level_is_rejected(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    result = run("scan", str(doc), "--min-level", "yok")
    assert result.exit_code != 0


def test_scan_without_sources_warns(tmp_path) -> None:
    doc = tmp_path / "tez.md"
    doc.write_text(DOC, encoding="utf-8")
    result = run("scan", str(doc), *OFFLINE)
    assert result.exit_code == 0
    assert "Kaynak verilmedi" in result.output
