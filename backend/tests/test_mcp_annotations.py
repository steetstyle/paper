"""A tool that reaches the network must not claim it stays at home.

MCP annotations are not decoration. ``openWorldHint: false`` tells a client the
tool touches no external entity, and clients use exactly that to decide whether a
call can be auto-approved without asking the person holding the keyboard.

Two tools said so while reaching out to arxiv.org:

    search_arxiv   ArxivSearchService(container.arxiv, ...) → service.search(...)
    get_paper      container.arxiv.get_paper(arxiv_id)

The cost was concrete rather than theoretical: ten ``get_paper`` calls in one
session, every one auto-approved as a local read, until arXiv answered 429 nine
times. The code already had the honest annotation — ``_READ_ONLY_REMOTE`` — and
``ask_paper_corpus`` was using it.

This test reads the module rather than a hand-kept list, because a list is exactly
what would have let the two drift apart in the first place: a tool added next month
with a network call would not be on it. New tools are covered by construction.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator

import pytest

SERVER = pathlib.Path(__file__).resolve().parents[1] / "app" / "mcp" / "server.py"

#: The handles that actually perform network I/O.
#:
#: ``container.arxiv`` is the ArXiv HTTP client and ``container.fetcher`` is the
#: shared one, so a tool that touches either reaches the open world however it then
#: uses them — ``ask_paper_corpus`` builds a service from the same client.
#:
#: Deliberately *not* the substring ``arxiv``: ``paper.arxiv_id`` is a column and
#: ``papers.get_by_arxiv_id`` is a repository method, both firmly local. Matching
#: those flagged ``list_assets``, ``list_references`` and ``list_project_papers``,
#: which would have made the guard cry wolf on four tools instead of two.
OUTBOUND_HANDLES = frozenset({("container", "arxiv"), ("container", "fetcher")})

#: Names that are network machinery wherever they appear.
#:
#: ``ArxivSearchService`` and ``build_ingestion_service`` are here for the indirect
#: route: a tool can reach arxiv.org without ever naming a client, by asking the
#: container for a service that holds one. ``ingest_paper`` did exactly that, which
#: is why a detector that only looked for ``container.arxiv`` would have called it
#: local and left the lie in place.
OUTBOUND_NAMES = frozenset({"HttpFetcher", "httpx", "ArxivSearchService", "build_ingestion_service"})


def _tools() -> Iterator[tuple[str, str | None, ast.AsyncFunctionDef]]:
    """Every registered tool: name, annotation constant, definition."""
    tree = ast.parse(SERVER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        name = annotation = None
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "attr", getattr(decorator.func, "id", ""))
                == "tool"
            ):
                continue
            for keyword in decorator.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    name = keyword.value.value
                elif keyword.arg == "annotations":
                    annotation = _annotation_constant(keyword.value)
        if name:
            yield name, annotation, node


def _annotation_constant(value: ast.expr) -> str | None:
    """``_ann(**_READS_LOCAL)`` → ``"_READS_LOCAL"``."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        for keyword in value.keywords:
            if isinstance(keyword.value, ast.Name):
                return keyword.value.id
    return None


def _reaches_outward(node: ast.AsyncFunctionDef) -> bool:
    """Whether the body touches the network, judged on the AST not on the text."""
    for child in ast.walk(node):
        # `container.arxiv.get_paper(...)` produces an Attribute for the chain and
        # one for the call; both are worth looking at, so accept either form.
        if isinstance(child, ast.Name) and child.id in OUTBOUND_NAMES:
            return True
        if not isinstance(child, ast.Attribute):
            continue
        chain: list[str] = []
        current: ast.expr = child
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.append(current.id)
        ordered = tuple(reversed(chain))
        if ordered in OUTBOUND_HANDLES:
            return True
        if any(part in OUTBOUND_NAMES for part in ordered):
            return True
    return False


class TestNetworkToolsAreLabelledAsSuch:
    def test_the_audit_actually_finds_the_network_tools(self) -> None:
        """A guard that finds nothing guards nothing.

        If this fails, the detection has stopped matching the code it was written
        for — and every other test here would pass while the labels rotted.
        """
        found = {name for name, _, node in _tools() if _reaches_outward(node)}
        assert "search_arxiv" in found, "detection no longer sees search_arxiv"
        assert "get_paper" in found, "detection no longer sees get_paper"

    @pytest.mark.parametrize(
        ("name", "annotation", "node"),
        [
            (name, annotation, node)
            for name, annotation, node in _tools()
            if _reaches_outward(node)
        ],
        ids=lambda value: getattr(value, "name", None) or str(value),
    )
    def test_a_tool_that_reaches_outward_is_not_labelled_local(
        self, name: str, annotation: str | None, node: ast.AsyncFunctionDef
    ) -> None:
        assert annotation != "_READS_LOCAL", (
            f"{name} reaches outside the process but is annotated _READS_LOCAL, "
            f"which tells a client it may auto-approve the call without asking"
        )

    @pytest.mark.parametrize(
        ("name", "annotation", "node"),
        [
            (name, annotation, node)
            for name, annotation, node in _tools()
            if _reaches_outward(node) and annotation and annotation.startswith("_WRITES")
        ],
        ids=lambda value: getattr(value, "name", None) or str(value),
    )
    def test_a_writer_that_downloads_says_it_reaches_outward(
        self, name: str, annotation: str | None, node: ast.AsyncFunctionDef
    ) -> None:
        """The same promise, for the tools that write.

        Ingestion pulls PDFs and HTML off arxiv.org. Labelling it closed-world tells
        a client the call touches nothing outside the process, and some will run it
        without asking — which is a lot to say yes to on someone's behalf when it
        means downloading and writing.
        """
        assert annotation == "_WRITES_IDEMPOTENT_REMOTE", (
            f"{name} writes and downloads but is annotated {annotation}"
        )

    def test_the_known_network_tools_say_so(self) -> None:
        """Named rather than only implied, so the fix is legible in a diff."""
        labels = {name: annotation for name, annotation, _ in _tools()}
        assert labels["search_arxiv"] == "_READ_ONLY_REMOTE"
        assert labels["get_paper"] == "_READ_ONLY_REMOTE"
        assert labels["ingest_paper"] == "_WRITES_IDEMPOTENT_REMOTE"
        assert labels["import_papers_to_project"] == "_WRITES_IDEMPOTENT_REMOTE"

    def test_a_purely_local_writer_is_not_relabelled(self) -> None:
        """``create_project`` writes a database row and downloads nothing. Marking
        it open-world would train the reader to distrust the label."""
        labels = {name: annotation for name, annotation, _ in _tools()}
        assert labels["create_project"] == "_WRITES_IDEMPOTENT"

    def test_purely_local_reads_keep_their_label(self) -> None:
        """The other direction: a genuinely local tool must not be relabelled, or
        the fix stops meaning anything."""
        labels = {name: annotation for name, annotation, _ in _tools()}
        for name in ("read_chunks", "read_markdown", "read_section", "status", "list_papers"):
            assert labels[name] == "_READS_LOCAL", name

    def test_no_tool_is_registered_twice(self) -> None:
        names = [name for name, _, _ in _tools()]
        duplicates = {n for n in names if names.count(n) > 1}
        assert duplicates == set(), f"registered more than once: {duplicates}"