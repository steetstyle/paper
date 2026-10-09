"""Search-query rendering and Atom feed parsing."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from paper_app.clients.arxiv.exceptions import ArxivParseError
from paper_app.clients.arxiv.parser import parse_entry, parse_feed
from paper_app.clients.arxiv.query import build_search_query
from paper_app.domain.models import SearchQuery


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


class TestGetPaperFallback:
    """`get_paper` must resolve old-style ids, and not give up on the first error.

    Every id here is real and was measured against the live API. ArXiv is
    inconsistent for the old-style (`archive/YYMMNNNN`) form: `id:` in a
    `search_query` matches nothing at all, `cond-mat/0305062v1` answers HTTP 500
    while its versionless form returns the paper, and `solv-int/9712001v2`
    returns nothing because v2 was never published.
    """

    @staticmethod
    def _client(*, bodies: list, settings=None):  # noqa: ANN205
        from paper_app.clients.arxiv.client import ArxivClient
        from paper_app.config import get_settings

        payload = iter(bodies)

        class Fake:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def get_text(self, url, params=None, **kw):  # noqa: ANN001, ANN003
                self.calls.append(dict(params or {}))

                class R:
                    status_code = 200
                    content = next(payload)
                    headers: dict = {}

                return R()

        fetcher = Fake()
        settings = settings or get_settings().arxiv
        return ArxivClient(settings, fetcher), fetcher

    @staticmethod
    def _entry(arxiv_id: str, *, authors: int = 2) -> str:
        names = "".join(
            f"<author><name>Author {i}</name></author>" for i in range(authors)
        )
        return (
            "<entry>"
            f"<id>http://arxiv.org/abs/{arxiv_id}</id>"
            f"<title>Paper {arxiv_id}</title>"
            f"<summary>An abstract of some length.</summary>"
            f"{names}"
            "</entry>"
        )

    @pytest.mark.asyncio
    async def test_old_style_id_resolves_through_id_list(self) -> None:
        from paper_app.clients.arxiv.parser import render_feed_xml

        client, fetcher = self._client(
            bodies=[render_feed_xml([self._entry("cond-mat/0404680v1")]).encode()]
        )
        metadata = await client.get_paper("cond-mat/0404680v1")
        # arxiv_id is the versionless canonical form; version carries the rest.
        assert metadata.arxiv_id == "cond-mat/0404680"
        assert metadata.versioned_id == "cond-mat/0404680v1"
        assert metadata.abstract
        assert len(metadata.authors) == 2
        # The request shape is the fix, not just the outcome.
        assert fetcher.calls[0]["id_list"] == "cond-mat/0404680v1"
        assert "search_query" not in fetcher.calls[0]

    @pytest.mark.asyncio
    async def test_a_500_falls_back_to_the_versionless_id(self) -> None:
        """`cond-mat/0305062v1` 500s; `cond-mat/0305062` returns the paper."""
        from paper_app.clients.arxiv.parser import render_feed_xml
        from paper_app.infra.http import HttpError

        error_body = render_feed_xml([self._entry("cond-mat/0305062v1")]).encode()
        ok_body = render_feed_xml([self._entry("cond-mat/0305062v4")]).encode()

        class Flaky:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def get_text(self, url, params=None, **kw):  # noqa: ANN001, ANN003
                self.calls.append(dict(params or {}))
                if len(self.calls) == 1:
                    raise HttpError("boom", status_code=500)

                class R:
                    status_code = 200
                    content = ok_body
                    headers: dict = {}

                return R()

        from paper_app.clients.arxiv.client import ArxivClient
        from paper_app.config import get_settings

        fetcher = Flaky()
        client = ArxivClient(get_settings().arxiv, fetcher)
        metadata = await client.get_paper("cond-mat/0305062v1")
        assert metadata.versioned_id == "cond-mat/0305062v4"
        assert fetcher.calls[0]["id_list"] == "cond-mat/0305062v1"
        assert fetcher.calls[1]["id_list"] == "cond-mat/0305062"
        assert error_body  # keeps the fixture honest

    @pytest.mark.asyncio
    async def test_an_unpublished_version_falls_back_too(self) -> None:
        """`solv-int/9712001v2` was never published; v1 exists."""
        from paper_app.clients.arxiv.parser import render_feed_xml

        empty = render_feed_xml([], total=0).encode()
        found = render_feed_xml([self._entry("solv-int/9712001v1")]).encode()
        client, fetcher = self._client(bodies=[empty, found])
        metadata = await client.get_paper("solv-int/9712001v2")
        assert metadata.versioned_id == "solv-int/9712001v1"
        assert [c["id_list"] for c in fetcher.calls] == [
            "solv-int/9712001v2",
            "solv-int/9712001",
        ]

    @pytest.mark.asyncio
    async def test_no_version_means_no_second_request(self) -> None:
        from paper_app.clients.arxiv.parser import render_feed_xml

        client, fetcher = self._client(
            bodies=[render_feed_xml([self._entry("cond-mat/0404680v1")]).encode()]
        )
        await client.get_paper("cond-mat/0404680v1")
        assert len(fetcher.calls) == 1

    @pytest.mark.asyncio
    async def test_a_404_is_not_swallowed(self) -> None:
        """Only a 500 is retried; a genuine 404 must surface."""
        from paper_app.infra.http import HttpError

        class Failing:
            async def get_text(self, url, params=None, **kw):  # noqa: ANN001, ANN003
                raise HttpError("gone", status_code=404)

        from paper_app.clients.arxiv.client import ArxivClient
        from paper_app.config import get_settings

        client = ArxivClient(get_settings().arxiv, Failing())
        with pytest.raises(HttpError):
            await client.get_paper("cond-mat/0404680v1")

