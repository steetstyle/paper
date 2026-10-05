"""Search-query rendering and Atom feed parsing."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from app.clients.arxiv.exceptions import ArxivParseError
from app.clients.arxiv.parser import parse_entry, parse_feed
from app.clients.arxiv.query import build_search_query
from app.domain.models import SearchQuery


@pytest.mark.parametrize(
    "query,expected",
    [
        (SearchQuery(all_terms=("transformer",)), "all:transformer"),
        (SearchQuery(title_terms=("attention",)), "ti:attention"),
        (SearchQuery(author_terms=("vaswani",)), "au:vaswani"),
        (SearchQuery(abstract_terms=("neural networks",)), 'abs:"neural networks"'),
        (SearchQuery(category_terms=("cs.LG",)), "cat:cs.LG"),
        (SearchQuery(title_terms=("a", "b")), "(ti:a OR ti:b)"),
        (SearchQuery(title_terms=("transformer",), category_terms=("cs.LG",)),
         "ti:transformer AND cat:cs.LG"),
        (SearchQuery(raw='ti:"agent" AND cat:cs.AI'), 'ti:"agent" AND cat:cs.AI'),
        (SearchQuery(id_list=("2401.01234", "2401.01235")),
         "(id:2401.01234 OR id:2401.01235)"),
    ],
)
def test_build_search_query(query: SearchQuery, expected: str) -> None:
    assert build_search_query(query) == expected


def test_search_query_is_empty() -> None:
    assert SearchQuery().is_empty()
    assert not SearchQuery(all_terms=("x",)).is_empty()


def test_parse_feed(feed_xml: str) -> None:
    page = parse_feed(feed_xml)
    assert page.total_results == 1
    assert len(page) == 1

    metadata = page.hits[0].metadata
    assert metadata.arxiv_id == "1706.03762"
    assert metadata.versioned_id == "1706.03762v5"
    assert metadata.version == 5
    assert metadata.title == "Attention Is All You Need"
    assert metadata.primary_category == "cs.CL"
    assert "cs.LG" in metadata.categories
    assert metadata.doi == "10.5555/3295222.3295349"
    assert [a.name for a in metadata.authors] == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    assert metadata.pdf_url.endswith("/pdf/1706.03762v5")
    assert metadata.published_at is not None and metadata.published_at.year == 2017
    assert "transformer" in metadata.abstract.lower() or "attention" in metadata.abstract.lower()


def test_parse_feed_multiple(feed_xml_multi: str) -> None:
    page = parse_feed(feed_xml_multi)
    assert page.total_results == 3
    assert len(page.hits) == 3
    assert len({hit.metadata.arxiv_id for hit in page.hits}) == 3


def test_parse_feed_rejects_non_feed() -> None:
    with pytest.raises(ArxivParseError, match="expected <feed>"):
        parse_feed("<html><body>not a feed</body></html>")
    with pytest.raises(ArxivParseError, match="invalid Atom XML"):
        parse_feed("<feed><unclosed>")


def test_html_url_proposed_from_version(feed_xml: str) -> None:
    page = parse_feed(feed_xml, html_base_url="https://arxiv.org/html")
    assert page.hits[0].metadata.html_url == "https://arxiv.org/html/1706.03762v5"


def test_parse_entry_without_authors() -> None:
    entry_xml = """<entry xmlns:arxiv="http://arxiv.org/schemas/atom"
                         xmlns="http://www.w3.org/2005/Atom">
      <id>http://arxiv.org/abs/2401.00001v1</id>
      <title>Bare Minimum</title>
      <summary>Only the required fields.</summary>
    </entry>"""
    hit = parse_entry(ET.fromstring(entry_xml))
    assert hit.metadata.authors == ()
    assert hit.metadata.categories == ()
    assert hit.metadata.primary_category is None