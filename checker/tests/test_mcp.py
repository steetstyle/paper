"""MCP server: tools are driven through ``server.call_tool``.

Same entry point the JSON-RPC layer uses, so schema conversion and result
encoding are exercised rather than bypassed. No test downloads weights: the
model-backed tools are called with the signals switched off, and the injection
point (``ScanRunner``'s ratio service) is covered separately.
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import pytest

from checker_app.mcp import server as mcp_server

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

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
"""

SOURCE = """# Kaynak

Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. Model
eğitimi üç ayrı koşul altında titizlikle yürütülmüştür ve sonuçlar kaydedilmiştir.
"""


def _decode(result: Any) -> dict:
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured.get("result", structured) if isinstance(structured, dict) else structured
    text = "".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )
    return json.loads(text)


def call(tool: str, /, **arguments: Any) -> dict:
    """Invoke a tool the way the protocol layer does."""
    return _decode(anyio.run(mcp_server.server.call_tool, tool, arguments))


@pytest.fixture
def thesis(tmp_path):
    path = tmp_path / "tez.md"
    path.write_text(THESIS, encoding="utf-8")
    return str(path)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "kaynak.md"
    path.write_text(SOURCE, encoding="utf-8")
    return str(path)


@pytest.fixture
def thesis_dir(tmp_path, thesis, source):
    """A directory holding both files, for the ``--ref-dir`` style tools."""
    (tmp_path / "tez.md").rename(tmp_path / "tez.txt")
    (tmp_path / "kaynak.md").rename(tmp_path / "kaynak.txt")
    return str(tmp_path / "tez.txt"), str(tmp_path)


# ------------------------------------------------------------------ discovery
def test_server_advertises_every_tool() -> None:
    async def _list() -> list[str]:
        tools = await mcp_server.server.list_tools()
        return [t.name for t in tools]

    names = anyio.run(_list)
    assert names == [
        "checker_segment",
        "checker_scan",
        "checker_similarity",
        "checker_compliance",
        "checker_sentences",
        "checker_matches",
        "checker_compare_revisions",
        "checker_calibrate",
        "checker_doctor",
        "checker_patchwork",
        "checker_references",
    ]


def test_tools_declare_read_only_annotations() -> None:
    async def _annotations() -> list[tuple[str, bool | None]]:
        tools = await mcp_server.server.list_tools()
        return [
            (t.name, getattr(t.annotations, "read_only_hint", None)) for t in tools
        ]

    for name, read_only in anyio.run(_annotations):
        assert read_only is True, f"{name} okuma-only olmalı"


def test_disclaimer_resource_is_served() -> None:
    async def _read() -> str:
        # ``ReadResourceContents`` exposes ``content``; there is no ``.text``.
        contents = await mcp_server.server.read_resource("checker://disclaimer")
        return "".join(getattr(item, "content", "") or "" for item in contents)

    text = anyio.run(_read)
    assert "TAHMİN" in text
    assert "RAID" in text


def test_disclaimer_carries_the_turkish_detector_failure() -> None:
    """The measured Turkish false-positive rates, which is this tool's core case."""
    async def _read() -> str:
        contents = await mcp_server.server.read_resource("checker://disclaimer")
        return "".join(getattr(item, "content", "") or "" for item in contents)

    text = anyio.run(_read)
    assert "%89 AI" in text, "Altıntop'ın Justdone sonucu eksik"
    assert "%73.25" in text, "diller arası salınım eksik"
    assert "%5.84" in text, "yayımlanmış Türkçe FPR eksik"
    assert "%95.21" in text, "akademik alan sonucu eksik"
    assert "10.56493/nkusbmyo.1866431" in text


def test_the_scan_caveats_include_the_turkish_fpr(thesis) -> None:
    payload = call(
        "checker_scan",
        path=thesis,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=1,
    )
    caveats = payload["verdict"]["caveats"]
    assert "%89 AI" in caveats["turkish_fpr"]
    assert caveats["turkish_fpr_source"].startswith("Altıntop (2026)")
    # Distinct from the German-lecture study: merging them would produce a
    # number that means nothing.
    assert "%57" in caveats["human_baseline"]
    assert "%57" not in caveats["turkish_fpr"]


def test_compliance_reports_the_yok_guide_text(tmp_path) -> None:
    """A rule row that says "disclose AI use" without saying which uses are
    forbidden is not actionable."""
    path = tmp_path / "tez.md"
    path.write_text(
        "# Giriş\n\nVeri temizliği gerçekleştirilmiştir.\n",
        encoding="utf-8",
    )
    payload = call("checker_compliance", path=str(path))
    guide = payload["compliance"]["yok_guide"]
    assert guide["numeric_threshold"] is None
    assert guide["mentions_thesis"] is False
    assert "hipotez üretimi" in guide["forbidden"]
    assert "çeviri" in guide["permitted"]
    item = next(i for i in payload["compliance"]["items"] if i["code"] == "ai_disclosure")
    assert "YASAK" in item["detail"]


def test_similarity_carries_the_turkish_ai_presence_distribution(thesis, source) -> None:
    """A Turkish reference for the AI share, the way similarity has one."""
    payload = call("checker_similarity", path=thesis, refs=[source])
    presence = payload["similarity"]["turkish_ai_presence"]
    assert presence["sample_size"] == 204
    assert presence["mean_percent"] == 20.0
    assert presence["below_20_percent_share"] == pytest.approx(0.598)
    # AI concentrates in the opening; the method section is nearly clean.
    assert presence["section_rates"]["method"] == pytest.approx(0.029)
    assert "tez değil" in presence["caveat"]


def test_disclaimer_carries_the_measured_guards() -> None:
    """The two rules that keep a reader from over-reading the output."""
    async def _read() -> str:
        contents = await mcp_server.server.read_resource("checker://disclaimer")
        return "".join(getattr(item, "content", "") or "" for item in contents)

    text = anyio.run(_read)
    assert "%26.85" in text, "düzeltme derecesi uyarısı eksik"
    assert "%57" in text and "%64" in text, "insan tabanı eksik"
    assert "%20" in text, "raporlama tabanı eksik"
    assert "Waterloo" in text, "kurumsal kapanışlar eksik"


def test_instructions_state_the_limits() -> None:
    text = mcp_server.INSTRUCTIONS
    assert "cited_and_quoted" in text
    assert "6.4%" in text
    assert "never proof" in text or "never" in text


# ----------------------------------------------------------------- tool calls
def test_doctor_reports_models_and_the_english_only_classifier() -> None:
    payload = call("checker_doctor", language="tr")
    assert payload["language"] == "tr"
    assert "likelihood_ratio" in payload["models"]
    assert payload["models"]["classifier_usable"] is False
    assert any("İngilizce" in note for note in payload["notes"])
    assert payload["operating_point"]["min_ai_words"] >= 1


def test_segment_needs_no_models(thesis) -> None:
    payload = call("checker_segment", path=thesis, max_items=5)
    assert payload["sentences"] >= 6
    assert payload["language"] == "tr"
    first = payload["items"][0]
    assert first["section_role"] in {"abstract", "front_matter", "other", "heading"}
    assert "character_range" in first
    assert payload["items"][0]["character_range"][0] == 0


def test_segment_marks_unscored_short_sentences(thesis) -> None:
    payload = call("checker_segment", path=thesis, max_items=30)
    roles = {item["section_role"] for item in payload["items"]}
    assert "abstract" in roles
    assert "introduction" in roles
    assert "references" in roles


def test_scan_without_models_still_reports_the_plagiarism(thesis, source) -> None:
    payload = call(
        "checker_scan",
        path=thesis,
        refs=[source],
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=5,
    )
    verdict = payload["verdict"]
    assert verdict["risk"] in {"none", "low", "medium", "high", "critical"}
    assert payload["sources_used"] == [source.rsplit("/", 1)[-1]]
    assert payload["document"]["sentences"] >= 6
    # The verbatim method paragraph must be found.
    assert verdict["matched_sentences"] >= 1
    assert payload["disclaimer"]


def test_scan_findings_carry_locations(thesis) -> None:
    payload = call(
        "checker_scan",
        path=thesis,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=3,
    )
    for finding in payload["findings"]:
        assert "character_range" in finding
        assert "location" in finding
        assert "risk" in finding
        assert finding["section_role"]


def test_similarity_reports_institutional_numbers(thesis, source) -> None:
    payload = call("checker_similarity", path=thesis, refs=[source])
    similarity = payload["similarity"]
    assert similarity["filters"]["description"]
    assert similarity["stats"]["total_words"] > 0
    assert similarity["incl_quotes_percent"] >= 0.0
    assert "excl_quotes_percent" in similarity
    assert "largest_single_source_percent" in similarity
    assert similarity["baseline"]["mean_percent"] == 28.7
    assert len(similarity["institutional_bands"]) >= 4
    assert all(band["url"].startswith("http") for band in similarity["institutional_bands"])


def test_similarity_excludes_the_references_section_by_default(thesis, source) -> None:
    payload = call("checker_similarity", path=thesis, refs=[source])
    stats = payload["similarity"]["stats"]
    assert stats["reference_words"] > 0, "kaynakça bölümü sayımdan çıkarılmalı"
    assert stats["matched_words_in_references"] == 0


def test_similarity_without_sources_explains_itself(thesis) -> None:
    payload = call("checker_similarity", path=thesis)
    assert "error" in payload


def test_patchwork_reports_the_shape_and_its_chance_baseline(thesis, source) -> None:
    payload = call("checker_patchwork", path=thesis, refs=[source])
    pw = payload["patchwork"]
    # The verbatim method paragraph is one reused run, so exactly one tile.
    assert pw["tile_count"] >= 1
    assert pw["largest_tile_words"] >= 5
    assert pw["score"] == pytest.approx(pw["tile_count"] / (pw["document_words"] - 1), rel=1e-3)
    # The one chance baseline in this literature that was actually measured.
    assert pw["chance_baseline"] == 0.15
    assert pw["above_chance"] is (pw["score"] >= 0.15)
    assert payload["chance_baselines_measured"]["GIT (greedy identifier tiles, math)"] == 0.15
    # HyPlag's GIT >= 0.15 was measured over 1,000,000 random pairs; publishing it
    # is the point, since every institutional percentage is a locally chosen number.
    assert payload["chance_baselines_measured"]["Encoplot (shared character 16-grams)"] == 0.06
    assert "COPE" in pw["shape"] or pw["shape"] == "yok"


def test_patchwork_needs_sources(thesis) -> None:
    assert "error" in call("checker_patchwork", path=thesis)


def test_references_flags_a_malformed_doi_and_a_future_year(tmp_path) -> None:
    path = tmp_path / "tez.md"
    path.write_text(
        "# Kaynakça\n\n"
        "Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri. Eğitim Bilimleri "
        "Dergisi, 12(3), 45-61.\n\n"
        "Smith, J. ve Jones, P. (2019). Deep learning for assessment. Journal of "
        "Learning Analytics. doi:10.1234/jla.2019.5678\n\n"
        "Zhang, W. (2023). Attention mechanisms revisited. Neural Networks. "
        "doi:10.9999\n\n"
        "Doe, A. (2099). Future perspectives on tutoring. Journal of Applied "
        "Research.\n",
        encoding="utf-8",
    )
    payload = call("checker_references", path=str(path), show_all=True)
    audit = payload["references"]
    assert audit["references_scanned"] == 4
    by_flags = {tuple(e["flags"]) for e in audit["entries"]}
    assert ("doi_yazilmis_ama_gecersiz", "kalici_kimlik_yok") in by_flags
    assert ("kalici_kimlik_yok", "gelecek_yil") in by_flags
    # A legitimate print-only reference is a note, never a review: no DOI alone
    # does not make a Turkish reference suspicious.
    assert audit["risk_counts"]["review"] == 2
    assert any(e["identifier"] == "10.1234/jla.2019.5678" for e in audit["entries"])


def test_references_can_show_the_clean_entries(tmp_path) -> None:
    path = tmp_path / "tez.md"
    path.write_text(
        "# Kaynakça\n\n"
        "Smith, J. ve Jones, P. (2019). Deep learning for assessment. Journal of "
        "Learning Analytics. doi:10.1234/jla.2019.5678\n",
        encoding="utf-8",
    )
    flagged_only = call("checker_references", path=str(path))
    everything = call("checker_references", path=str(path), show_all=True)
    assert flagged_only["references"]["risk_counts"]["ok"] == 1
    assert flagged_only["references"]["entries"] == []
    assert len(everything["references"]["entries"]) == 1


def test_references_says_so_when_there_is_no_bibliography(tmp_path) -> None:
    path = tmp_path / "tez.md"
    path.write_text("# Giriş\n\nVeri temizliği gerçekleştirilmiştir.\n", encoding="utf-8")
    assert "error" in call("checker_references", path=str(path))


def test_scan_does_not_claim_a_document_verdict_below_the_floor(tmp_path, source) -> None:
    """The 20% gate: per-sentence findings survive, the document AI claim does not.

    The fixture is deliberately non-generic Turkish prose, which is what a real
    thesis introduction looks like. Stylometry alone puts these sentences at
    0.34-0.46, below the 0.55 medium threshold, so the AI share stays under the
    floor even though two of them are uncited matches.
    """
    path = tmp_path / "insan.md"
    path.write_text(
        "# 1. GİRİŞ\n\n"
        "Bu tezde odaklandığım soru şuydu: sınıf içinde kendi cümlesini kurma\n"
        "cesareti dağılan öğrenciler, yazarken hangi kaygıyla karşılaşıyor? Önce kendi\n"
        "derslerimdeki notlarımı taradım, sonra 2019'dan bu yana yayımlanmış otuz iki\n"
        "çalışmayı okudum. Çoğu ölçüm aracının öğrenciye ne hissettirdiğini ölçmüyordu;\n"
        "hepsi doğru cevabı arıyordu.\n\n"
        "Görüşmeleri altı hafta sürdü ve yirmi üç öğrenciyle yapıldı. İlk üç görüşme\n"
        "işe yaramadı; sorularım çok resmiydi ve öğrenciler bana sınav cevabı verdi.\n"
        "Bundan sonra not tutmayı bıraktım ve açık uçlu sormaya çalıştım.\n",
        encoding="utf-8",
    )
    payload = call(
        "checker_scan",
        path=str(path),
        refs=[source],
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=5,
    )
    verdict = payload["verdict"]
    assert verdict["below_ai_reporting_floor"] is True
    assert 0.0 <= verdict["ai_share"] < 0.20
    assert verdict["document_ai_claim"] is None, "tabanın altında belge AI iddiası olmamalı"
    assert "%20" in verdict["document_ai_claim_note"]
    # The sentences are still listed, with their locations - a floor, not a
    # silent clean bill of health.
    assert payload["findings"]
    assert "character_range" in payload["findings"][0]


def test_scan_issues_a_document_claim_above_the_floor(thesis) -> None:
    """The gate is a floor, not a blanket refusal: generic prose does get scored."""
    payload = call(
        "checker_scan",
        path=thesis,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=3,
    )
    verdict = payload["verdict"]
    assert verdict["below_ai_reporting_floor"] is False
    assert verdict["ai_share"] >= 0.20
    assert verdict["document_ai_claim"] == verdict["ai_score"]
    assert verdict["document_ai_claim"] > 0.0


def test_verdict_carries_the_document_level_context(thesis) -> None:
    """Numbers a reader could over-read must arrive with their caveat attached."""
    payload = call(
        "checker_scan",
        path=thesis,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=1,
    )
    context = payload["verdict"]["style_context"]
    # Word count alone reaches ROC-AUC 0.68, so the regime is named.
    assert context["length_confound"]["measured"]["word_count_only_auroc"] == 0.68
    assert context["length_confound"]["regime"] in {"cok_kisa", "calisma_kosuluna_yakin", "uzun"}
    assert context["length_confound"]["excluded"]
    # Lexical richness is not a verdict; the measured sign table travels with it.
    assert context["lexical_richness"] is None or "bağlam olarak raporlanır" in (
        context["lexical_richness"]["reading"]
    )
    assert context["discourse"]["measured_direction"]["attribution"] == "human"


def test_verdict_carries_the_degree_caveat_and_the_human_baseline(thesis) -> None:
    payload = call(
        "checker_scan",
        path=thesis,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=1,
    )
    caveats = payload["verdict"]["caveats"]
    assert "%26.85" in caveats["degree"]
    assert "%57" in caveats["human_baseline"]
    assert "0.608" in caveats["human_baseline"]


def test_compliance_flags_a_missing_disclosure(thesis, source) -> None:
    payload = call("checker_compliance", path=thesis, refs=[source])
    items = {item["code"]: item for item in payload["compliance"]["items"]}
    assert items["ai_disclosure"]["status"] == "required"
    assert "YÖK" in items["ai_disclosure"]["source"]
    assert payload["compliance"]["notes"]


def test_compliance_accepts_a_disclosed_thesis(tmp_path, source) -> None:
    text = THESIS + (
        "\n# Yöntem\n\nBu bölümde yapay zekâ araçlarından yararlanılmıştır ve "
        "çıktılar yazar tarafından doğrulanmıştır.\n"
    )
    path = tmp_path / "tez.md"
    path.write_text(text, encoding="utf-8")
    payload = call("checker_compliance", path=str(path), refs=[source])
    items = {item["code"]: item for item in payload["compliance"]["items"]}
    assert items["ai_disclosure"]["status"] == "ok"


def test_matches_are_attributed(thesis, source) -> None:
    payload = call("checker_matches", path=thesis, refs=[source], unattributed_only=True)
    assert payload["total_matching"] >= 1
    match = payload["matches"][0]
    assert match["citation_status"] in {
        "not_cited_or_quoted",
        "missing_citation",
        "missing_quotation",
    }
    assert match["character_range"][1] > match["character_range"][0]
    assert "attribution_counts" in payload


def test_matches_can_include_cited_ones(thesis, source) -> None:
    everything = call(
        "checker_matches", path=thesis, refs=[source], unattributed_only=False
    )
    uncited = call(
        "checker_matches", path=thesis, refs=[source], unattributed_only=True
    )
    assert everything["total_matching"] >= uncited["total_matching"]


def test_matches_needs_sources(thesis) -> None:
    payload = call("checker_matches", path=thesis)
    assert "error" in payload


def test_sentences_pages_and_filters(thesis) -> None:
    payload = call(
        "checker_sentences",
        path=thesis,
        min_level="none",
        limit=3,
        offset=0,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
    )
    assert payload["returned"] == 3
    assert payload["total_matching"] >= 3
    second = call(
        "checker_sentences",
        path=thesis,
        min_level="none",
        limit=3,
        offset=3,
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
    )
    assert second["sentences"][0]["index"] > payload["sentences"][-1]["index"]


def test_compare_revisions(tmp_path) -> None:
    left = tmp_path / "v1.md"
    right = tmp_path / "v2.md"
    left.write_text("Birinci cümle burada bulunmaktadır. İkinci cümle şuradadır.", encoding="utf-8")
    right.write_text("Birinci cümle burada bulunmaktadır. Üçüncü cümle eklendi.", encoding="utf-8")
    payload = call("checker_compare_revisions", left_path=str(left), right_path=str(right))
    assert payload["counts"]["equal"] >= 1
    assert payload["total_changed"] >= 1
    assert payload["changed"][0]["op"] in {"insert", "replace"}


def test_scan_skips_the_document_itself(thesis, thesis_dir) -> None:
    path, directory = thesis_dir
    payload = call(
        "checker_scan",
        path=path,
        ref_dirs=[directory],
        use_ratio=False,
        use_perplexity=False,
        use_classifier=False,
        min_level="none",
        top=2,
    )
    assert path.rsplit("/", 1)[-1] in " ".join(payload["sources_skipped"])


def test_calibrate_reports_a_missing_corpus(tmp_path) -> None:
    empty = tmp_path / "bos"
    empty.mkdir()
    payload = call("checker_calibrate", human_dir=str(empty))
    assert "error" in payload