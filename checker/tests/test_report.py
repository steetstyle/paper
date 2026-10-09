"""Reports: JSON shape, Markdown readability, annotated text, diffs."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from checker_app.config import CheckerSettings
from checker_app.domain.enums import DiffOp, MatchKind, RiskLevel, Severity, Signal
from checker_app.domain.models import (
    ClassifierStats,
    Finding,
    MatchSpan,
    PerplexityStats,
    PlagiarismStats,
    StyleHit,
    StyleStats,
)
from checker_app.services.report import diff_markdown, to_annotated_text, to_json, to_markdown
from checker_app.services.runner import ScanRequest, ScanRunner

TEXT = (
    "# Giriş\n\n"
    "Ayrıca, bu çalışmada yöntemin bütün ayrıntıları sunulmaktadır. "
    "Veri temizliği gerçekleştirilmiştir ve standart bir protokol kullanılmıştır."
)


def build_report():
    """A real scan with the model signals off, plus an injected plagiarism hit."""
    runner = ScanRunner(CheckerSettings())
    report = runner.scan(
        ScanRequest(
            path="tez.md",
            text=TEXT,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
            use_plagiarism=False,
        )
    )

    match = MatchSpan(
        kind=MatchKind.EXACT,
        source_id="s1",
        source_name="kaynak.md",
        sentence_indices=(1,),
        doc_char_start=20,
        doc_char_end=60,
        source_char_start=0,
        source_char_end=40,
        matched_words=18,
        ratio=1.0,
        snippet="Veri temizliği gerçekleştirilmiştir",
    )
    target = report.sentences[1]
    patched = replace(
        target,
        style=StyleStats(
            word_count=target.word_count,
            char_count=len(target.text),
            hits=(
                StyleHit(
                    code="discourse_marker",
                    severity=Severity.MEDIUM,
                    weight=0.16,
                    detail="geçiş kalıbı",
                    evidence="Ayrıca",
                ),
            ),
        ),
        perplexity=PerplexityStats(
            perplexity=18.4,
            mean_nll=2.9,
            token_count=24,
            log10_perplexity=1.26,
            z_score=-1.4,
            model="gpt2",
        ),
        classifier=ClassifierStats(p_ai=0.81, model="clf"),
        plagiarism=PlagiarismStats(
            containment=0.8,
            longest_match_words=18,
            best_source="kaynak.md",
            matches=(match,),
        ),
        ai_score=0.62,
        plagiarism_score=0.71,
        risk_score=0.78,
        risk=RiskLevel.HIGH,
        confidence=1.0,
        findings=target.findings
        + (
            Finding(
                code="match_exact",
                signal=Signal.PLAGIARISM,
                severity=Severity.HIGH,
                score=0.97,
                message="kaynak.md içinden 18 kelime birebir kopyalanmış görünüyor.",
                evidence="karakter 20-60",
            ),
        ),
    )
    sentences = list(report.sentences)
    sentences[1] = patched
    return replace(report, sentences=tuple(sentences))


def test_json_has_a_stable_shape() -> None:
    payload = json.loads(to_json(build_report()))
    assert payload["schema_version"] == 1
    assert set(payload) >= {
        "disclaimer",
        "document",
        "verdict",
        "models",
        "degraded_signals",
        "sections",
        "sentences",
    }
    first = payload["sentences"][0]
    assert set(first) >= {"index", "text", "location", "scores", "style", "findings"}
    assert first["location"]["line_start"] >= 1
    assert first["location"]["label"]


def test_json_includes_matches_with_offsets() -> None:
    payload = json.loads(to_json(build_report()))
    matches = payload["sentences"][1]["plagiarism"]["matches"]
    assert matches[0]["source_name"] == "kaynak.md"
    assert matches[0]["doc_char_start"] == 20
    assert matches[0]["matched_words"] == 18


def test_markdown_shows_location_and_reasons() -> None:
    report = build_report()
    markdown = to_markdown(report, min_level=RiskLevel.NONE, include_all=True)
    assert "satır" in markdown
    assert "kaynak.md" in markdown
    assert "perplexity" in markdown
    assert "### #1" in markdown
    assert report.disclaimer[:20] in markdown


def test_markdown_can_filter_and_limit() -> None:
    report = build_report()
    only_critical = to_markdown(report, min_level=RiskLevel.CRITICAL)
    assert "### #" not in only_critical
    limited = to_markdown(report, min_level=RiskLevel.NONE, limit=1)
    assert limited.count("### #") == 1


def test_annotated_text_puts_a_badge_after_each_sentence() -> None:
    annotated = to_annotated_text(build_report(), TEXT)
    assert "⟦#0" in annotated
    assert "intihal %71" in annotated
    assert "ppl 18" in annotated
    for line in annotated.splitlines():
        if line.strip():
            # Badges may sit mid-line when several sentences share a line.
            assert "⟧" in line, line
    assert annotated.count("⟦#") == len(build_report().sentences)
    # Removing the badges must return the document byte for byte.
    stripped = re.sub(r"⟦#[^⟧]*⟧", "", annotated)
    assert stripped == TEXT


def test_diff_markdown_counts_operations() -> None:
    runner = ScanRunner(CheckerSettings())
    diffs = runner.diff(
        "Birinci cümle burada. İkinci cümle orada.",
        "Birinci cümle burada. Üçüncü cümle eklendi.",
    )
    markdown = diff_markdown(diffs)
    assert "yeni" in markdown
    assert diffs[-1].op in {DiffOp.INSERT, DiffOp.REPLACE}
    assert any(d.op is DiffOp.EQUAL for d in diffs)
    assert any(d.right_location is not None for d in diffs)
    assert any(d.left_location is not None for d in diffs)


def test_runner_degrades_when_signals_are_disabled() -> None:
    report = ScanRunner(CheckerSettings()).scan(
        ScanRequest(
            path="x.md",
            text=TEXT,
            use_perplexity=False,
            use_classifier=False,
            use_ratio=False,
        )
    )
    assert all(s.perplexity is None for s in report.sentences)
    assert all(s.classifier is None for s in report.sentences)
    assert report.models == {}


def test_report_helpers() -> None:
    report = build_report()
    ordered = report.sorted_by_risk()
    assert ordered[0].risk_score >= ordered[-1].risk_score
    assert report.sections_of(["Giriş"]) is not None
    assert len(report.flagged(RiskLevel.HIGH)) >= 1
