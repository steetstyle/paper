"""Source loading: files, directories, literals, extraction, tokenisation."""

from __future__ import annotations

from checker_app.config import CheckerSettings, PlagiarismSettings
from checker_app.services.sources import SourceLoader


def test_load_document_with_metadata(tmp_path) -> None:
    path = tmp_path / "tez.md"
    path.write_text("# Başlık\n\nBirinci cümle burada.\n", encoding="utf-8")
    loader = SourceLoader(CheckerSettings())
    loaded = loader.load_document(path)
    assert loaded.text.startswith("# Başlık")
    assert loaded.kind == "file"
    assert loaded.meta["suffix"] == ".md"


def test_missing_file_raises(tmp_path) -> None:
    loader = SourceLoader(CheckerSettings())
    try:
        loader.load_document(tmp_path / "yok.md")
    except FileNotFoundError:
        return
    raise AssertionError("FileNotFoundError bekleniyordu")


def test_unknown_reference_is_treated_as_literal_text() -> None:
    loader = SourceLoader(CheckerSettings())
    loaded = loader.load_reference("bu bir kaynak metni değil ama olabilir")
    assert loaded.kind == "text"
    assert loaded.text == "bu bir kaynak metni değil ama olabilir"


def test_directory_loading_is_filtered_and_sorted(tmp_path) -> None:
    (tmp_path / "b.md").write_text("ikinci kaynak metni burada yer alıyor.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("birinci kaynak metni burada yer alıyor.", encoding="utf-8")
    (tmp_path / "gorsel.png").write_bytes(b"\x89PNG")

    loader = SourceLoader(CheckerSettings())
    loaded = loader.load_directory(tmp_path)
    assert [item.path.rsplit("/", 1)[-1] for item in loaded] == ["a.txt", "b.md"]

    sources = loader.to_sources(loaded)
    assert sources[0].name == "a.txt"
    assert len(sources[0].tokens) > 0
    assert sources[0].tokens[0].char_start == 0


def test_source_tokens_are_folded_for_comparison(tmp_path) -> None:
    path = tmp_path / "k.md"
    path.write_text("Göztepe'de IŞIK ölçüldü ve İZMİR karşılaştırıldı.", encoding="utf-8")
    loader = SourceLoader(CheckerSettings())
    source = loader.to_sources([loader.load_document(path)])[0]
    norms = [t.norm for t in source.tokens]
    assert "goztepede" in norms
    assert "ısık" in norms  # dotless I survives folding, on purpose


def test_token_offsets_point_into_the_source_text(tmp_path) -> None:
    text = "Veri temizliği gerçekleştirilmiştir ve protokol kullanılmıştır."
    path = tmp_path / "k.txt"
    path.write_text(text, encoding="utf-8")
    loader = SourceLoader(CheckerSettings())
    source = loader.to_sources([loader.load_document(path)])[0]
    for token in source.tokens:
        assert text[token.char_start : token.char_end] == token.text


def test_html_is_extracted_to_text(tmp_path) -> None:
    path = tmp_path / "s.html"
    path.write_text(
        "<html><body><p>Merhaba dünya</p><script>var x=1;</script></body></html>",
        encoding="utf-8",
    )
    loader = SourceLoader(CheckerSettings())
    loaded = loader.load_document(path)
    assert "Merhaba dünya" in loaded.text
    assert "var x" not in loaded.text


def test_max_chars_is_applied_to_sources(tmp_path) -> None:
    settings = CheckerSettings(plagiarism=PlagiarismSettings(max_source_chars=1000))
    path = tmp_path / "uzun.txt"
    path.write_text("kelime " * 400, encoding="utf-8")
    loader = SourceLoader(settings)
    source = loader.to_sources([loader.load_document(path)])[0]
    assert len(source.text) == 1000


def test_source_ids_are_stable_and_unique(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("bir", encoding="utf-8")
    (tmp_path / "b.txt").write_text("iki", encoding="utf-8")
    loader = SourceLoader(CheckerSettings())
    sources = loader.to_sources(loader.load_directory(tmp_path))
    ids = [s.source_id for s in sources]
    assert ids[0] != ids[1]
    assert ids[0] == loader.to_sources(loader.load_directory(tmp_path))[0].source_id
