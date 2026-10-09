"""Report rendering: JSON, Markdown and an annotated copy of the text.

Three shapes for three audiences:

* ``to_json`` - stable ``schema_version``ed dict for CI, diffing and storage
* ``to_markdown`` - a reviewable document: verdict, sections, then one row per
  sentence with its location, scores and reasons
* ``to_annotated_text`` - the original text with per-sentence badges, so the
  findings can be read *in place* in the prose rather than in a table
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from checker_app.domain.enums import DiffOp, RiskLevel
from checker_app.domain.models import DocumentReport, SentenceDiff, SentenceReport
from checker_app.domain.sections import role_label

__all__ = [
    "to_json",
    "to_markdown",
    "to_annotated_text",
    "diff_markdown",
    "similarity_markdown",
    "compliance_markdown",
]

_LEVEL_MARK = {
    RiskLevel.NONE: "OK",
    RiskLevel.LOW: "düşük",
    RiskLevel.MEDIUM: "orta",
    RiskLevel.HIGH: "yüksek",
    RiskLevel.CRITICAL: "kritik",
}
_OP_MARK = {
    DiffOp.EQUAL: "=",
    DiffOp.INSERT: "+",
    DiffOp.DELETE: "-",
    DiffOp.REPLACE: "~",
}


def to_json(report: DocumentReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=indent)


def to_markdown(
    report: DocumentReport,
    *,
    min_level: RiskLevel = RiskLevel.LOW,
    limit: int | None = None,
    include_all: bool = False,
) -> str:
    lines: list[str] = []
    verdict = (
        f"# Metin risk raporu — {report.path}\n\n"
        f"- **Risk:** {_LEVEL_MARK[report.risk]} ({report.risk_score:.2f})\n"
        f"- **AI skoru:** {report.ai_score:.2f} · "
        f"**İntihal skoru:** {report.plagiarism_score:.2f}\n"
        f"- **Cümle:** {report.sentence_count} · **Kelime:** {report.word_count} · "
        f"**Sayfa:** {report.page_count}\n"
        f"- **Dil:** {report.language} (güven {report.language_confidence:.2f})\n"
        f"- **İşaretli cümle:** {report.flagged_sentences} · "
        f"**yüksek riskli:** {report.high_risk_sentences} · "
        f"**eşleşen:** {report.matched_sentences}\n"
    )
    if report.mean_perplexity is not None:
        verdict += (
            f"- **Ortalama perplexity:** {report.mean_perplexity:.1f} · "
            f"**burstiness (ppl): {report.perplexity_burstiness} · "
            f"**burstiness (uzunluk): {report.length_burstiness}\n"
        )
    if report.models:
        verdict += (
            "- **Modeller:** " + ", ".join(f"{k}=`{v}`" for k, v in report.models.items()) + "\n"
        )
    if report.degraded_signals:
        verdict += "- **Çalışmayan sinyaller:** " + "; ".join(report.degraded_signals) + "\n"
    if report.below_ai_reporting_floor:
        verdict += (
            f"- **AI kapsamı %{report.ai_share * 100:.1f}** — raporlama tabanının "
            "(%20) altında; belge düzeyinde AI hükmü verilmiyor, yalnız cümle "
            "düzeyi bulgular raporlanıyor.\n"
        )
    lines.append(verdict)
    lines.append(f"> {report.disclaimer}\n")
    # The two measured limits belong in the file a committee actually reads,
    # not only in the JSON: the degree caveat and the human baseline are what
    # stop "3 sentences flagged" from becoming an accusation.
    if report.degree_caveat:
        lines.append(f"> **{report.degree_caveat}**\n")
    if report.human_baseline:
        lines.append(f"> **{report.human_baseline}**\n")
    if report.turkish_fpr:
        lines.append(f"> **{report.turkish_fpr}**\n")

    if report.sections:
        lines.append("\n## Bölümler\n")
        lines.append("| Bölüm | Cümle | Kelime | AI | İntihal | Yüksek risk |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for section in report.sections:
            lines.append(
                f"| {section.title} | {len(section.sentence_indices)} | {section.word_count} | "
                f"{section.ai_score:.2f} | {section.plagiarism_score:.2f} | "
                f"{section.high_risk_count} |"
            )

    selected = [s for s in report.sentences if include_all or s.risk.at_least(min_level)]
    selected.sort(key=lambda s: (-s.risk_score, s.index))
    if limit:
        selected = selected[:limit]

    lines.append(f"\n## Cümle cümle sonuçlar ({len(selected)}/{len(report.sentences)})\n")
    if not selected:
        lines.append("_Eşiği aşan cümle yok._")
    for sentence in selected:
        lines.append(_sentence_block(sentence))

    if report.similarity is not None:
        lines.append(_similarity_block(report))
    if report.integrity is not None and report.integrity.findings:
        lines.append("\n## Metin bütünlüğü\n")
        for finding in report.integrity.findings:
            example = (" — " + ", ".join(finding.examples[:3])) if finding.examples else ""
            lines.append(f"- **{finding.code}** ×{finding.count}: {finding.detail}{example}")

    if report.sources:
        lines.append("\n## Karşılaştırılan kaynaklar\n")
        for source in report.sources:
            lines.append(
                f"- `{source.get('name')}` ({source.get('kind')}, {source.get('words')} kelime)"
            )

    lines.append(
        f"\n---\n_Üretim süresi {report.duration_seconds:.2f}s · "
        f"şema sürümü {report.to_dict()['schema_version']}_\n"
    )
    return "\n".join(lines)


def _similarity_block(report: DocumentReport) -> str:
    """The institutional-style similarity block, with its provenance."""
    report_similarity = report.similarity
    lines = ["\n## Benzerlik (kurumsal filtrelerle)\n"]
    lines.append(
        f"- **Alıntılar dahil:** {report_similarity.similarity_incl_quotes:.2f}% "
        "(Turnitin'in ana rakamı)"
    )
    lines.append(f"- **Alıntılar hariç:** {report_similarity.similarity_excl_quotes:.2f}%")
    lines.append(
        f"- **Kaynakça hariç:** {report_similarity.similarity_excl_references:.2f}% · "
        f"**kaynakça ve alıntılar hariç:** "
        f"{report_similarity.similarity_excl_references_quotes:.2f}%"
    )
    largest = report_similarity.largest_single_source
    if largest:
        lines.append(f"- **En yüksek tek kaynak:** {largest.share_percent:.2f}% (`{largest.name}`)")
    stats = report_similarity.stats
    lines.append(
        f"- Örtüşen {stats.matched_words} kelime / {stats.total_words} · "
        f"{stats.match_count} eşleşme, {stats.cases} vaka "
        f"(granülarite {stats.granularity}) · filtre: {report_similarity.filters}"
    )
    lines.append("\n| Atıf durumu | Eşleşme |")
    lines.append("|---|---:|")
    for key, count in report_similarity.attribution.items():
        lines.append(f"| {key} | {count} |")
    baseline = report_similarity.baseline
    if report_similarity.english_baseline is not None:
        english = report_similarity.english_baseline
        lines.append(
            f"\nÖlçülen **İngilizce** referans: doktora tezi ortalaması "
            f"**{english.mean_percent}%** ± {english.sd_percent} "
            f"(n={english.sample_size}, {english.field}). Türkçe referansın "
            "yaklaşık üçte biri; İngilizce bir belgeye Türkçe eşik uygulamak "
            "normali bir tezin büyük kısmını işaretlerdi.\n"
        )
        lines.append(f"\n{english.reading(report_similarity.similarity_incl_quotes)}")
        lines.append(
            f"\n- Ölçülmüş **optimal eşik %{english.optimal_cutoff_percent:.0f}**: "
            f"duyarlılık %{english.cutoff_sensitivity * 100:.1f}, "
            f"özgüllük %{english.cutoff_specificity * 100:.1f}, AUC {english.cutoff_auc}. "
            "Kurumsal %25-30 tavanı bir *politik* sınırdır, gözlenen normal değil."
        )
        lines.append(
            f"\n- Doğrulanmış intihal vakalarının %{english.non_native_share_of_cases * 100:.0f}'si "
            "İngilizcenin resmî dil olmadığı ülkelerden. Ana dilini İngilizce "
            "olmayan yazarın metni düşük leksik çeşitlilik gösterebilir ve yine "
            "tamamen insani olabilir."
        )
    lines.append(
        f"\nÖlçülen Türkçe tez dağılımı: ortalama **{baseline.mean_percent}%** "
        f"± {baseline.sd_percent} (n={baseline.sample_size}, {baseline.field}); "
        f"Türkçe dilde {baseline.turkish_percent}%, İngilizce dilde "
        f"{baseline.english_percent}%. Kaynak: {baseline.source}. "
        f"{baseline.caveat}"
    )
    coverage = report_similarity.language_coverage
    if coverage is not None:
        lines.append(f"\n**Dil kapsamı:** {coverage.reading()}")
        lines.append(f"\n> {coverage.evidence}")
    cross_lingual = report.cross_lingual
    if cross_lingual is not None:
        payload = cross_lingual.to_dict()
        lines.append(
            f"\n**Çapraz dil aktarımı:** {payload['clusters_found']} küme "
            f"(yüksek {payload['confidence_counts']['high']}, "
            f"orta {payload['confidence_counts']['medium']}, "
            f"gözden geçir {payload['confidence_counts']['review']})"
        )
        for cluster in payload["clusters"][:10]:
            lines.append(
                f"  - [{cluster['confidence']}] {cluster['source']}: "
                + ", ".join(cluster["anchors"])
            )
        lines.append(f"\n{cross_lingual.reading()}")
        lines.append(f"\n> {cross_lingual.evidence}\n> {cross_lingual.caveat}")
    presence = report_similarity.ai_presence
    lines.append(f"\n{presence.to_dict()['reading']}")
    lines.append("\n| Kurum | Eşikler | Durum |")
    lines.append("|---|---|---|")
    for band in report_similarity.bands:
        thresholds = band["thresholds"]
        bits = ", ".join(
            f"{label} {value:g}%"
            for label, value in (
                ("alıntılar hariç", thresholds["total_excl_quotes_percent"]),
                ("alıntılar dahil", thresholds["total_incl_quotes_percent"]),
                ("tek kaynak", thresholds["single_source_percent"]),
            )
            if value is not None
        )
        detail = "; ".join(band["issues"]) if band["issues"] else "eşik aşılmadı"
        reason = band.get("no_threshold_reason")
        if reason:
            detail = reason
        thresholds_text = bits or "sayısal eşik yok"
        lines.append(f"| {band['institution']} | {thresholds_text} | {detail} |")
    return "\n".join(lines)


def _sentence_block(sentence: SentenceReport) -> str:
    head = (
        f"### #{sentence.index} · {_LEVEL_MARK[sentence.risk]} "
        f"(risk {sentence.risk_score:.2f} · AI {sentence.ai_score:.2f} · "
        f"intihal {sentence.plagiarism_score:.2f} · güven {sentence.confidence:.2f})\n\n"
    )
    role = ""
    if sentence.section_role.value != "other":
        role = f" · {role_label(sentence.section_role)}"
    meta = f"`{sentence.location.label()}`{role} · {sentence.location.citation()}\n\n"
    body = f"> {sentence.snippet(320)}\n\n"
    details: list[str] = []
    if sentence.perplexity:
        details.append(
            f"- perplexity: {sentence.perplexity.perplexity:.1f} "
            f"(token {sentence.perplexity.token_count}, "
            f"bağlam {sentence.perplexity.context_tokens})"
        )
    if sentence.ratio:
        details.append(
            f"- likelihood-ratio: {sentence.ratio.score:.3f} "
            f"({sentence.ratio.observer_model} {sentence.ratio.observer_perplexity:.0f} / "
            f"{sentence.ratio.performer_model} {sentence.ratio.performer_perplexity:.0f})"
        )
    if sentence.classifier:
        details.append(
            f"- sınıflandırıcı: %{sentence.classifier.p_ai * 100:.0f} "
            f"(`{sentence.classifier.model}`)"
        )
    if sentence.plagiarism and sentence.plagiarism.has_match:
        for match in sentence.plagiarism.matches[:3]:
            details.append(
                f"- eşleşme ({match.kind.value}): `{match.source_name}` · "
                f"{match.matched_words} kelime · "
                f"karakter {match.doc_char_start}-{match.doc_char_end}"
            )
            if match.snippet:
                details.append(f"  - metin: _{match.snippet[:160]}_")
    for finding in sentence.findings[:6]:
        evidence = f" — _{finding.evidence}_" if finding.evidence else ""
        details.append(f"- {finding.message}{evidence}")
    tail = "\n".join(details) + "\n\n" if details else ""
    return head + meta + body + tail


def to_annotated_text(report: DocumentReport, text: str) -> str:
    """The original document with an inline badge after every analysed sentence.

    Badges are inserted at the sentence character offsets, so the original
    layout - paragraphs, blank lines, code fences - is preserved byte for byte.
    """
    badges = {s.location.char_end: _badge(s) for s in report.sentences}
    if not badges:
        return text
    out: list[str] = []
    for offset, character in enumerate(text):
        # char_end is exclusive, so the badge goes before the character that
        # follows the sentence, not after it.
        badge = badges.pop(offset, None)
        if badge:
            out.append(badge)
        out.append(character)
    # The last sentence can end exactly at the end of the document.
    for badge in badges.values():
        out.append(badge)
    return "".join(out)


def _badge(sentence: SentenceReport) -> str:
    mark = _LEVEL_MARK[sentence.risk]
    parts = [f"⟦#{sentence.index} {mark}"]
    parts.append(f"AI %{sentence.ai_percent}")
    if sentence.plagiarism and sentence.plagiarism.has_match:
        parts.append(f"intihal %{sentence.plagiarism_percent}")
    if sentence.ratio:
        parts.append(f"oran {sentence.ratio.score:.2f}")
    elif sentence.perplexity:
        parts.append(f"ppl {sentence.perplexity.perplexity:.0f}")
    reasons = [f.code for f in sentence.findings if f.severity.value in {"medium", "high"}]
    if reasons:
        parts.append("sebep: " + ",".join(dict.fromkeys(reasons)))
    return " | ".join(parts) + "⟧"


def diff_markdown(diffs: Sequence[SentenceDiff]) -> str:
    """Sentence-level diff of two revisions, as Markdown."""
    lines = ["# Cümle düzeyi karşılaştırma\n"]
    counts = {op: 0 for op in DiffOp}
    for diff in diffs:
        counts[diff.op] += 1
    lines.append(
        f"- eşit: {counts[DiffOp.EQUAL]} · yeni: {counts[DiffOp.INSERT]} · "
        f"silinen: {counts[DiffOp.DELETE]} · değişen: {counts[DiffOp.REPLACE]}\n"
    )
    for diff in diffs:
        if diff.op is DiffOp.EQUAL:
            continue
        mark = _OP_MARK[diff.op]
        location = diff.left_location or diff.right_location
        where = f"`{location.label()}`" if location else ""
        if diff.op is DiffOp.INSERT:
            lines.append(f"{mark} yeni cümle {where}\n\n> {diff.right_text}\n")
        elif diff.op is DiffOp.DELETE:
            lines.append(f"{mark} silinen cümle {where}\n\n> {diff.left_text}\n")
        else:
            lines.append(f"{mark} değişen cümle {where}\n")
            lines.append(f"- önceki: _{diff.left_text}_")
            lines.append(f"- yeni: _{diff.right_text}_\n")
    return "\n".join(lines)


def similarity_markdown(report: DocumentReport) -> str:
    """Just the similarity section, for `checker similarity`."""
    if report.similarity is None:
        return (
            "# Benzerlik raporu\n\n"
            "Kaynak verilmediği için benzerlik hesaplanmadı. "
            "`--ref`, `--ref-dir` veya `--ref-url` ile kaynak ekleyin.\n"
        )
    return "# Benzerlik raporu\n" + _similarity_block(report) + "\n"


def compliance_markdown(report: DocumentReport, compliance) -> str:
    """The YOK/institutional checklist as a reviewable document."""
    from checker_app.services.compliance import ComplianceReport  # noqa: PLC0415

    if not isinstance(compliance, ComplianceReport):
        compliance = build_compliance_report(report)
    status_mark = {
        "ok": "uygun",
        "note": "bilgi",
        "review": "gözden geçir",
        "required": "gerekli",
    }
    lines = [
        f"# Uyum kontrol listesi — {report.path}\n",
        f"- **Risk:** {report.risk.value} · **AI:** {report.ai_score:.2f} · "
        f"**intihal:** {report.plagiarism_score:.2f}",
        f"- **Gerekli maddeler:** {compliance.blocking}\n",
        "## Maddeler\n",
        "| Durum | Kural | Bulgu | Nerede | Ne yapılacak |",
        "|---|---|---|---|---|",
    ]
    for item in compliance.items:
        lines.append(
            f"| {status_mark.get(item.status, item.status)} | {item.rule} | "
            f"{item.evidence} | {item.where} | {item.action or '-'} |"
        )
    lines.append("\n### Kaynaklar\n")
    for item in compliance.items:
        if item.source:
            lines.append(f"- **{item.code}**: {item.source}")
    if compliance.ai_heavy_sections:
        lines.append("\n## AI yoğun bölümler\n")
        for section in compliance.ai_heavy_sections:
            lines.append(
                f"- **{section['section']}** ({section['role']}): "
                f"AI skoru {section['ai_score']:.2f}"
            )
            for row in section["sentences"]:
                lines.append(f"  - #{row['index']} ({row['ai_score']:.2f}) {row['snippet']}")
    if compliance.uncited_matches:
        lines.append("\n## Atıfsız eşleşmeler\n")
        for row in compliance.uncited_matches:
            lines.append(
                f"- cümle #{row['sentence']} (`{row['location']}`) ↔ `{row['source']}` · "
                f"{row['matched_words']} kelime ({row['kind']})\n  - _{row['snippet']}_"
            )
    if compliance.notes:
        lines.append("\n## Notlar\n")
        for note in compliance.notes:
            lines.append(f"- {note}")
    lines.append(f"\n---\n{report.disclaimer}\n")
    return "\n".join(lines)


def build_compliance_report(report: DocumentReport, *, raw_text: str = ""):
    """Late import to keep the reporter free of service dependencies."""
    from checker_app.services.compliance import (  # noqa: PLC0415
        build_compliance_report as _build,
    )

    return _build(report, raw_text=raw_text)
