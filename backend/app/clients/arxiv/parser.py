"""ArXiv Atom (RFC 4287) feed parser.

The API returns a single ``feed`` document whose ``entry`` elements carry both
Dublin-Core and the ``arxiv:`` extension namespace. Parsing is namespace-aware
and tolerant of missing optional fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

from app.clients.arxiv.exceptions import ArxivParseError
from app.domain.enums import ArxivSortBy, ArxivSortOrder
from app.domain.ids import parse_arxiv_id, strip_version, url_for_html, url_for_pdf
from app.domain.models import (
    Author,
    PaperMetadata,
    SearchHit,
    SearchQuery,
    SearchResultPage,
)

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

_NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS, "opensearch": OPENSEARCH_NS}

_ENTRY_TITLE_FIELDS = (
    "{http://purl.org/dc/elements/1.1/}title",
    "{http://purl.org/dc/terms/}title",
    f"{{{ATOM_NS}}}title",
)


def _text(element: ET.Element | None, *paths: str) -> str | None:
    if element is None:
        return None
    for path in paths:
        node = element.find(path, _NS)
        if node is not None and node.text and node.text.strip():
            return _clean(node.text)
    return None


def _clean(value: str) -> str:
    return " ".join(value.split())


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_entry(entry: ET.Element, *, html_base_url: str | None = None) -> SearchHit:
    """Map one ``<entry>`` element onto :class:`SearchHit`."""
    raw_id = _text(entry, f"{{{ATOM_NS}}}id")
    if not raw_id:
        raise ArxivParseError("entry is missing an <id>")

    parsed = parse_arxiv_id(raw_id)
    arxiv_id = parsed.id
    version = parsed.version
    versioned_id = parsed.versioned

    title = _text(entry, *_ENTRY_TITLE_FIELDS)
    if not title:
        raise ArxivParseError(f"entry {versioned_id} is missing a title")

    abstract = _text(
        entry,
        "{http://purl.org/dc/elements/1.1/}summary",
        "{http://purl.org/dc/terms/}summary",
        f"{{{ATOM_NS}}}summary",
    ) or ""

    authors: list[Author] = []
    for node in entry.findall(f"{{{ATOM_NS}}}author"):
        name = _text(node, f"{{{ATOM_NS}}}name")
        if not name:
            continue
        affiliation_node = node.find(f"{{{ARXIV_NS}}}affiliation")
        affiliation = (
            _clean(affiliation_node.text)
            if affiliation_node is not None and affiliation_node.text
            else None
        )
        authors.append(Author(name=name, affiliation=affiliation))

    categories: list[str] = []
    for node in entry.findall(f"{{{ATOM_NS}}}category"):
        term = node.get("term")
        if term and term not in categories:
            categories.append(term)

    primary_node = entry.find(f"{{{ARXIV_NS}}}primary_category")
    primary_category = primary_node.get("term") if primary_node is not None else None
    if primary_category and primary_category not in categories:
        categories.insert(0, primary_category)

    doi = _text(entry, f"{{{ARXIV_NS}}}doi")
    comment = _text(entry, f"{{{ARXIV_NS}}}comment")
    journal_ref = _text(entry, f"{{{ARXIV_NS}}}journal_ref")

    links = _parse_links(entry)
    pdf_url = links.get("pdf") or url_for_pdf(arxiv_id, version)
    abs_url = links.get("abs") or f"https://arxiv.org/abs/{versioned_id}"

    # ArXiv HTML5 renderings exist from Dec 2023 onwards; the fetcher probes it.
    html_url = links.get("html")
    if html_url is None and version is not None:
        candidate = (html_base_url or "https://arxiv.org/html").rstrip("/") + f"/{versioned_id}"
        html_url = candidate

    metadata = PaperMetadata(
        arxiv_id=arxiv_id,
        versioned_id=versioned_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=tuple(authors),
        categories=tuple(categories),
        primary_category=primary_category,
        published_at=_parse_datetime(_text(entry, f"{{{ATOM_NS}}}published")),
        updated_at=_parse_datetime(_text(entry, f"{{{ATOM_NS}}}updated")),
        doi=doi,
        comment=comment,
        journal_ref=journal_ref,
        abs_url=abs_url,
        pdf_url=pdf_url,
        html_url=html_url,
        raw={"id": raw_id},
    )
    score = _text(entry, f"{{{ARXIV_NS}}}score")
    return SearchHit(metadata=metadata, relevance_score=float(score) if score else None)


def _parse_links(entry: ET.Element) -> dict[str, str]:
    links: dict[str, str] = {}
    for node in entry.findall(f"{{{ATOM_NS}}}link"):
        href = node.get("href")
        if not href:
            continue
        title = (node.get("title") or "").lower()
        rel = (node.get("rel") or "").lower()
        if title == "pdf" or href.lower().endswith(".pdf"):
            links["pdf"] = href
        elif title == "html" or "html" in title:
            links["html"] = href
        elif rel == "alternate" and "abs" in href:
            links["abs"] = href
    return links


def parse_feed(
    payload: bytes | str,
    *,
    query: SearchQuery | None = None,
    html_base_url: str | None = None,
) -> SearchResultPage:
    """Parse a full Atom feed into a :class:`SearchResultPage`."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ArxivParseError(f"invalid Atom XML: {exc}") from exc

    if root.tag != f"{{{ATOM_NS}}}feed":
        raise ArxivParseError(f"expected <feed>, got <{root.tag}>")

    total_node = root.find("opensearch:totalResults", _NS)
    total = int(total_node.text) if total_node is not None and total_node.text else 0

    start_node = root.find("opensearch:startIndex", _NS)
    start = int(start_node.text) if start_node is not None and start_node.text else 0

    per_page_node = root.find("opensearch:itemsPerPage", _NS)
    per_page = int(per_page_node.text) if per_page_node is not None and per_page_node.text else 0

    hits = tuple(
        parse_entry(entry, html_base_url=html_base_url)
        for entry in root.findall(f"{{{ATOM_NS}}}entry")
    )
    if total == 0 and hits:
        total = len(hits)
    if per_page == 0:
        per_page = len(hits)
    if start == 0 and query is not None:
        start = query.start

    return SearchResultPage(hits=hits, total_results=total, start=start, items_per_page=per_page)


def entry_xml_fixtures() -> dict[str, Any]:  # pragma: no cover - docs helper
    """Minimal, valid Atom sample used by the test-suite and docs."""
    return {
        "id": "http://arxiv.org/abs/1706.03762v5",
        "title": "Attention Is All You Need",
        "summary": "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks.",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "primary": "cs.CL",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2017-06-12T00:00:00Z",
        "updated": "2023-08-02T00:00:00Z",
    }


def render_entry_xml(spec: dict[str, Any]) -> str:
    """Render a feed entry from :func:`entry_xml_fixtures`-style data (tests)."""
    authors = "".join(
        f"<author><name>{name}</name></author>" for name in spec.get("authors", [])
    )
    categories = "".join(
        f'<category term="{term}" scheme="http://arxiv.org/schemas/atom" />'
        for term in spec.get("categories", [])
    )
    primary = spec.get("primary")
    primary_node = (
        f'<arxiv:primary_category xmlns:arxiv="{ARXIV_NS}" term="{primary}" scheme="http://arxiv.org/schemas/atom" />'
        if primary
        else ""
    )
    return f"""<entry>
  <id>{spec['id']}</id>
  <updated>{spec['updated']}</updated>
  <published>{spec['published']}</published>
  <title>{spec['title']}</title>
  <summary>{spec['summary']}</summary>
  {authors}
  <arxiv:doi xmlns:arxiv="{ARXIV_NS}">10.5555/3295222.3295349</arxiv:doi>
  {primary_node}
  {categories}
  <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate" type="text/html" />
  <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf" />
</entry>"""


def render_feed_xml(entries: list[str], *, total: int | None = None) -> str:
    """Wrap rendered entries into a complete Atom feed document (tests)."""
    total = total if total is not None else len(entries)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="{ATOM_NS}" xmlns:opensearch="{OPENSEARCH_NS}">
  <link href="http://arxiv.org/api/query" rel="self" type="application/atom+xml" />
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/fake</id>
  <updated>2024-01-01T00:00:00-05:00</updated>
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>{len(entries)}</opensearch:itemsPerPage>
  {''.join(entries)}
</feed>"""


__all__ = [
    "parse_feed",
    "parse_entry",
    "ArxivSortBy",
    "ArxivSortOrder",
    "strip_version",
    "url_for_html",
]