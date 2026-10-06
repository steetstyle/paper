"""ArXiv filter language: compilation, validation and client-side filtering.

Pure tests. Nothing here touches the network, so every assertion is about what
we hand ArXiv, not what ArXiv happens to return today.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from app.clients.arxiv.filters import SearchRequest, search_query_from
from app.clients.arxiv.query import build_search_params
from app.domain.filters import (
    ArxivField,
    ArxivFilter,
    ArxivOperator,
    FilterProblem,
    PostFilter,
    format_submitted_date,
    parse_submitted_date,
    quote_if_needed,
    validate_arxiv_query,
)
from app.domain.models import PaperMetadata, SearchQuery


def metadata(**overrides: Any) -> PaperMetadata:
    """A minimal paper, overridable per test."""
    base: dict[str, Any] = {
        "arxiv_id": "1706.03762",
        "versioned_id": "1706.03762v5",
        "version": 5,
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer.",
        "categories": ("cs.CL", "cs.LG"),
        "primary_category": "cs.CL",
        "published_at": datetime(2017, 6, 12, tzinfo=UTC),
        "updated_at": datetime(2023, 8, 2, tzinfo=UTC),
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "html_url": None,
        "doi": "10.1000/xyz",
        "journal_ref": "NeurIPS 2017",
    }
    base.update(overrides)
    return PaperMetadata(**base)


class TestFieldPrefixes:
    """Every prefix the ArXiv manual documents must compile."""

    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            ("ti", "ti:transformer"),
            ("au", "au:vaswani"),
            ("abs", "abs:attention"),
            ("co", "co:accepted"),
            ("jr", "jr:NeurIPS"),
            ("cat", "cat:cs.CL"),
            ("rn", "rn:LA-UR-"),
            ("all", "all:transformer"),
        ],
    )
    def test_each_prefix_compiles(self, prefix: str, expected: str) -> None:
        value = expected.split(":", 1)[1]
        built = ArxivFilter.from_values(fields={prefix: [value]})
        assert built.compile() == expected

    def test_readable_aliases_map_onto_prefixes(self) -> None:
        assert ArxivField.resolve("title") is ArxivField.TITLE
        assert ArxivField.resolve("TI") is ArxivField.TITLE
        assert ArxivField.resolve("journal") is ArxivField.JOURNAL_REF
        assert ArxivField.resolve("journal-ref") is ArxivField.JOURNAL_REF
        assert ArxivField.resolve("report_number") is ArxivField.REPORT_NUMBER

    def test_unknown_field_is_rejected_with_the_supported_list(self) -> None:
        with pytest.raises(FilterProblem) as excinfo:
            ArxivField.resolve("keyword")
        message = str(excinfo.value)
        assert "keyword" in message
        # The error has to be actionable: it names what is valid.
        assert "ti" in message and "cat" in message


class TestCompilation:
    def test_multiple_values_in_a_field_are_anded_by_default(self) -> None:
        built = ArxivFilter.from_values(fields={"title": ["a", "b"]})
        assert built.compile() == "ti:a AND ti:b"

    def test_or_combines_values_within_a_field(self) -> None:
        built = ArxivFilter.from_values(fields={"title": ["a", "b"]}, operator=ArxivOperator.OR)
        assert built.compile() == "ti:a OR ti:b"

    def test_operator_does_not_leak_across_fields(self) -> None:
        """`--op OR` means "within a field"; across fields ArXiv needs AND."""
        built = ArxivFilter.from_values(
            fields={"title": ["a", "b"], "author": ["ho"]}, operator=ArxivOperator.OR
        )
        assert built.compile() == "ti:a OR ti:b AND au:ho"

    def test_raw_clause_is_anded_in_not_or_ed_in(self) -> None:
        built = ArxivFilter.from_values(
            fields={"title": ["a", "b"]}, raw="cat:cs.CL", operator=ArxivOperator.OR
        )
        assert built.compile() == "cat:cs.CL AND ti:a OR ti:b"

    def test_multiword_terms_become_phrases(self) -> None:
        built = ArxivFilter.from_values(fields={"title": ["attention is all you need"]})
        assert built.compile() == 'ti:"attention is all you need"'

    def test_single_token_terms_are_not_quoted(self) -> None:
        assert quote_if_needed("transformer") == "transformer"
        assert quote_if_needed("cs.CL") == "cs.CL"
        assert quote_if_needed("LA-UR-23-1") == "LA-UR-23-1"

    def test_hand_written_expressions_are_not_requoted(self) -> None:
        """Re-quoting a caller's expression would silently change its meaning."""
        for expression in [
            "(ti:a OR ti:b)",
            'ti:"already quoted"',
            "cat:cs.CL",
            "submittedDate:[202301010000 TO 202312312359]",
        ]:
            assert quote_if_needed(expression) == expression

    def test_raw_expression_survives_byte_for_byte(self) -> None:
        expression = '(ti:"attention is all you need" OR ti:transformer) ANDNOT cat:cs.LG'
        built = ArxivFilter.from_values(raw=expression)
        assert built.compile() == expression

    def test_raw_and_structured_terms_are_both_kept(self) -> None:
        """Passing both means both: neither side is silently discarded."""
        built = ArxivFilter.from_values(fields={"title": ["transformer"]}, raw="cat:cs.CL")
        assert built.compile() == "cat:cs.CL AND ti:transformer"

    def test_is_empty_detects_absence_of_every_clause(self) -> None:
        assert ArxivFilter().is_empty
        assert ArxivFilter.from_values(fields={"title": ["  "]}).is_empty
        assert not ArxivFilter.from_values(fields={"title": ["x"]}).is_empty
        assert not ArxivFilter.from_values(raw="cat:cs.CL").is_empty
        assert not ArxivFilter.from_values(submitted_from=datetime(2024, 1, 1)).is_empty

    def test_id_prefix_is_refused_with_a_pointer_to_id_list(self) -> None:
        with pytest.raises(FilterProblem) as excinfo:
            ArxivFilter.from_values(fields={"id": ["1706.03762"]})
        assert "id_list" in str(excinfo.value)


class TestSubmittedDate:
    """`submittedDate` is evaluated by ArXiv, so the format has to be exact."""

    def test_formats_as_yyyymmddtttt_in_gmt(self) -> None:
        moment = datetime(2023, 5, 4, 13, 30, tzinfo=UTC)
        assert format_submitted_date(moment) == "202305041330"

    def test_converts_to_gmt_before_formatting(self) -> None:

        # 13:30 in a UTC+2 zone is 11:30 GMT, and ArXiv wants GMT.
        moment = datetime(2023, 5, 4, 13, 30, tzinfo=timezone(timedelta(hours=2)))
        assert format_submitted_date(moment) == "202305041130"

    def test_range_renders_with_to_separator(self) -> None:
        built = ArxivFilter.from_values(
            fields={"title": ["x"]},
            submitted_from=datetime(2023, 1, 1, tzinfo=UTC),
            submitted_to=datetime(2023, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert "submittedDate:[202301010000 TO 202312312359]" in built.compile()

    def test_open_ended_range_still_produces_both_bounds(self) -> None:
        """ArXiv needs both ends; an absent upper bound means 'until now'."""
        built = ArxivFilter.from_values(submitted_from=datetime(2024, 1, 1, tzinfo=UTC))
        clause = built.compile()
        assert clause.startswith("submittedDate:[202401010000 TO ")
        assert clause.endswith("]")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2023-05-04", datetime(2023, 5, 4, tzinfo=UTC)),
            ("20230504", datetime(2023, 5, 4, tzinfo=UTC)),
            ("202305041330", datetime(2023, 5, 4, 13, 30, tzinfo=UTC)),
            ("2023-05-04T13:30:00Z", datetime(2023, 5, 4, 13, 30, tzinfo=UTC)),
            ("2023-05-04T13:30:00+00:00", datetime(2023, 5, 4, 13, 30, tzinfo=UTC)),
        ],
    )
    def test_accepts_the_formats_a_user_is_likely_to_type(self, value, expected) -> None:  # noqa: ANN001
        assert parse_submitted_date(value) == expected

    def test_blank_and_none_mean_no_bound(self) -> None:
        assert parse_submitted_date(None) is None
        assert parse_submitted_date("") is None
        assert parse_submitted_date("   ") is None

    def test_unreadable_date_explains_the_accepted_formats(self) -> None:
        with pytest.raises(FilterProblem) as excinfo:
            parse_submitted_date("last tuesday")
        assert "YYYY-MM-DD" in str(excinfo.value)

    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        parsed = parse_submitted_date(datetime(2023, 5, 4))
        assert parsed is not None
        assert parsed.tzinfo is UTC


    def test_separator_survives_url_encoding(self) -> None:
        """The compiled clause must survive being put in a query string.

        ArXiv's manual writes `+TO+`, but in a URL `+` *is* a space, so any
        encoder emits `%2B` and ArXiv answers 500. A literal space becomes
        `%20`, which ArXiv parses. Verified against the live API.
        """
        import httpx

        built = ArxivFilter.from_values(
            fields={"title": ["transformer"]},
            submitted_from=datetime(2023, 1, 1, 6, tzinfo=UTC),
            submitted_to=datetime(2024, 1, 1, 6, tzinfo=UTC),
        )
        url = httpx.URL(
            "https://export.arxiv.org/api/query", params={"search_query": built.compile()}
        ).raw_path.decode()
        assert "%2B" not in url
        assert "submittedDate" in url
        assert "TO" in url

    def test_a_plus_separator_would_be_corrupted(self) -> None:
        """Documents the bug above so the space is not 'simplified' back."""
        import httpx

        plus = "submittedDate:[202301010600+TO+202401010600]"
        encoded = httpx.URL(
            "https://export.arxiv.org/api/query", params={"search_query": plus}
        ).raw_path.decode()
        assert "%2B" in encoded

class TestValidation:
    def test_correct_uppercase_operators_produce_no_warnings(self) -> None:
        result = validate_arxiv_query("ti:a AND ti:b OR ti:c ANDNOT ti:d")
        assert result.ok
        assert result.warnings == ()
        assert result.normalized is None

    def test_lowercase_operators_are_flagged(self) -> None:
        result = validate_arxiv_query("ti:a and ti:b")
        assert result.ok  # not an error, just likely wrong
        assert any("UPPERCASE" in w for w in result.warnings)

    def test_normalized_form_fixes_operator_case(self) -> None:
        result = validate_arxiv_query("ti:a and ti:b or ti:c")
        assert result.normalized == "ti:a AND ti:b OR ti:c"

    def test_normalized_is_only_offered_when_something_changed(self) -> None:
        assert validate_arxiv_query("ti:a AND ti:b").normalized is None

    def test_id_prefix_is_a_warning_not_an_error(self) -> None:
        """ArXiv still accepts it; it just resolves versions poorly."""
        result = validate_arxiv_query("id:1706.03762")
        assert result.ok
        assert any("id_list" in w for w in result.warnings)

    def test_unknown_prefix_is_an_error(self) -> None:
        result = validate_arxiv_query("keyword:transformer")
        assert not result.ok
        assert any("keyword" in e for e in result.errors)

    def test_unbalanced_open_paren_is_an_error(self) -> None:
        result = validate_arxiv_query("(ti:a AND ti:b")
        assert not result.ok
        assert any("(" in e for e in result.errors)

    def test_unbalanced_close_paren_is_an_error(self) -> None:
        result = validate_arxiv_query("ti:a)")
        assert not result.ok

    def test_unterminated_quote_is_an_error(self) -> None:
        result = validate_arxiv_query('ti:"attention is all you need')
        assert not result.ok
        assert any("quote" in e for e in result.errors)

    def test_parentheses_inside_quotes_do_not_unbalance(self) -> None:
        assert validate_arxiv_query('ti:"a (b" AND ti:c').ok

    def test_empty_query_is_an_error(self) -> None:
        assert not validate_arxiv_query("   ").ok

    def test_as_dict_is_serialisable(self) -> None:
        payload = validate_arxiv_query("ti:a and ti:b").as_dict()
        assert set(payload) == {"ok", "errors", "warnings", "normalized"}
        assert isinstance(payload["warnings"], list)


class TestSearchRequestBuilder:
    """`search_query_from` is the single entry point every surface funnels into."""

    def test_builds_the_compiled_query(self) -> None:
        request = search_query_from(
            raw="cat:cs.CL",
            fields={"title": ["transformer"], "category": ["cs.LG"]},
            submitted_from="2024-01-01",
        )
        compiled = request.filter.compile()
        assert compiled.startswith("cat:cs.CL AND ti:transformer AND cat:cs.LG")
        assert "submittedDate:[202401010000 TO " in compiled

    def test_unknown_field_name_is_rejected(self) -> None:
        with pytest.raises(FilterProblem) as excinfo:
            search_query_from(fields={"keyword": ["x"]})
        assert "keyword" in str(excinfo.value)

    def test_unknown_operator_is_rejected(self) -> None:
        with pytest.raises(FilterProblem) as excinfo:
            search_query_from(fields={"title": ["x"]}, operator="XOR")
        assert "AND" in str(excinfo.value)

    def test_unknown_post_filter_is_rejected(self) -> None:
        with pytest.raises(FilterProblem):
            search_query_from(fields={"title": ["x"]}, post={"has_pdfz": True})

    def test_comma_separated_category_list_is_accepted_as_one_value(self) -> None:
        """`--category cs.CL,cs.LG` reads naturally; ArXiv wants one term."""
        request = search_query_from(fields={"category": "cs.CL,cs.LG"})
        assert request.category == ("cs.CL,cs.LG",)

    def test_max_results_is_clamped_to_the_api_limit(self) -> None:
        assert search_query_from(fields={"title": ["x"]}, max_results=5000).max_results == 100
        assert search_query_from(fields={"title": ["x"]}, max_results=0).max_results == 1

    def test_negative_offset_is_clamped(self) -> None:
        assert search_query_from(fields={"title": ["x"]}, start=-5).start == 0

    def test_invalid_sort_falls_back_to_relevance(self) -> None:
        request = search_query_from(fields={"title": ["x"]}, sort_by="nonsense")
        assert request.sort_by == "relevance"

    def test_warnings_are_collected_on_the_request(self) -> None:
        request = search_query_from(raw="ti:a and ti:b")
        assert any("UPPERCASE" in w for w in request.warnings)

    def test_invalid_raw_query_raises_rather_than_warns(self) -> None:
        with pytest.raises(FilterProblem):
            search_query_from(raw="keyword:x")

    def test_to_search_query_carries_arxiv_request_params(self) -> None:
        request = search_query_from(
            fields={"title": ["x"]}, max_results=7, start=3, sort_order="ascending"
        )
        query = request.to_search_query()
        assert query.max_results == 7
        assert query.start == 3
        assert query.sort_order == "ascending"

    def test_id_list_only_produces_no_search_query(self) -> None:
        """ArXiv applies id_list and search_query with different logic, so
        sending an empty search_query alongside ids is not equivalent."""
        request = search_query_from(id_list=["1706.03762v5"])
        assert request.to_search_query().raw is None
        assert request.id_list == ("1706.03762v5",)

    def test_describe_reports_the_everything_asked_of_arxiv(self) -> None:
        described = search_query_from(
            fields={"title": ["x"]}, submitted_from="2024-01-01", post={"has_pdf": True}
        ).describe()
        assert described["filter"]["compiled"].startswith("ti:x")
        assert described["post_filter"]["has_pdf"] is True
        assert described["max_results"] == 10


class TestPostFilter:
    """Filters ArXiv's API has no way to express, applied to the response."""

    def test_has_pdf_keeps_only_entries_with_a_pdf(self) -> None:
        post = PostFilter(has_pdf=True)
        assert post.apply(metadata(pdf_url="https://x/1.pdf"), ingested=False)
        assert not post.apply(metadata(pdf_url=None), ingested=False)

    def test_no_pdf_is_the_inverse(self) -> None:
        post = PostFilter(has_pdf=False)
        assert post.apply(metadata(pdf_url=None), ingested=False)
        assert not post.apply(metadata(pdf_url="https://x/1.pdf"), ingested=False)

    def test_has_html(self) -> None:
        assert PostFilter(has_html=True).apply(metadata(html_url="https://x"), ingested=False)
        assert not PostFilter(has_html=True).apply(metadata(html_url=None), ingested=False)

    def test_has_doi_and_journal_ref(self) -> None:
        assert PostFilter(has_doi=True).apply(metadata(doi="10.1/x"), ingested=False)
        assert not PostFilter(has_doi=False).apply(metadata(doi="10.1/x"), ingested=False)
        assert PostFilter(has_journal_ref=True).apply(
            metadata(journal_ref="NeurIPS"), ingested=False
        )

    def test_categories_must_all_be_present(self) -> None:
        post = PostFilter(categories=("cs.CL", "cs.LG"))
        assert post.apply(metadata(categories=("cs.CL", "cs.LG", "stat.ML")), ingested=False)
        assert not post.apply(metadata(categories=("cs.CL")), ingested=False)

    def test_exclude_categories_drops_any_overlap(self) -> None:
        post = PostFilter(exclude_categories=("cs.LG",))
        assert not post.apply(metadata(categories=("cs.CL", "cs.LG")), ingested=False)
        assert post.apply(metadata(categories=("cs.CL")), ingested=False)

    def test_ingested_filters_on_local_corpus_membership(self) -> None:
        assert PostFilter(ingested=True).apply(metadata(), ingested=True)
        assert not PostFilter(ingested=True).apply(metadata(), ingested=False)
        assert PostFilter(ingested=False).apply(metadata(), ingested=False)

    def test_ingested_is_skipped_when_corpus_state_is_unknown(self) -> None:
        """Without corpus access we must not silently reject every paper."""
        assert PostFilter(ingested=True).apply(metadata(), ingested=None)

    def test_empty_post_filter_passes_everything(self) -> None:
        assert PostFilter().apply(metadata(pdf_url=None, html_url=None, doi=None), ingested=None)
        assert PostFilter().is_empty

    def test_multiple_clauses_are_combined_with_and(self) -> None:
        post = PostFilter(has_pdf=True, categories=("cs.CL",))
        assert post.apply(metadata(), ingested=False)
        assert not post.apply(metadata(pdf_url=None), ingested=False)
        assert not post.apply(metadata(categories=("cs.AI")), ingested=False)


class TestToDict:
    def test_request_describe_is_json_safe(self) -> None:
        import json

        request: SearchRequest = search_query_from(
            fields={"title": ["x"]}, submitted_from="2024-01-01"
        )
        json.dumps(request.describe())  # must not raise

    def test_filter_to_dict_includes_the_compiled_form(self) -> None:
        assert ArxivFilter.from_values(fields={"title": ["x"]}).to_dict()["compiled"] == "ti:x"

    def test_post_filter_to_dict_round_trips_flags(self) -> None:
        payload = PostFilter(has_pdf=False, exclude_categories=("cs.LG",)).to_dict()
        assert payload["has_pdf"] is False
        assert payload["exclude_categories"] == ["cs.LG"]


class TestSubmittedDateInRawExpressions:
    """`submittedDate` is legal inside search_query, so it must not be rejected.

    The ArXiv manual's own example uses it; flagging it as an unknown prefix
    would reject the documented syntax.
    """

    def test_the_manuals_example_validates(self) -> None:
        expression = "au:del_maestro AND submittedDate:[202301010600+TO+202401010600]"
        assert validate_arxiv_query(expression).ok

    def test_it_is_not_normalised_away(self) -> None:
        expression = "submittedDate:[202301010600+TO+202401010600]"
        assert validate_arxiv_query(expression).normalized is None

    def test_it_passes_through_untouched(self) -> None:
        expression = "au:del_maestro AND submittedDate:[202301010600+TO+202401010600]"
        assert ArxivFilter.from_values(raw=expression).compile() == expression

    def test_other_unknown_prefixes_are_still_rejected(self) -> None:
        assert not validate_arxiv_query("submittedYear:[2023 TO 2024]").ok


class TestBareTermsAreOredNotAnded:
    """ArXiv ORs adjacent bare words. Measured against the live API:

        `sheaf` 3491, `neural` 198684, `network` 335204
        `sheaf neural network`           -> 394907  (the union)
        `sheaf AND neural AND network`   ->     56
        `"sheaf neural network"`         ->     33

    A typo makes it worse: a term that matches nothing is dropped, not treated
    as a zero, so `sheaf neureal network` widens to 338593 instead of narrowing.
    """

    def test_multiple_bare_words_warn(self) -> None:
        result = validate_arxiv_query("sheaf neural network")
        assert any("ORs adjacent terms" in w for w in result.warnings)

    def test_the_warning_quotes_the_phrase_to_use(self) -> None:
        result = validate_arxiv_query("sheaf neural network")
        warning = next(w for w in result.warnings if "ORs adjacent" in w)
        assert '"sheaf neural network"' in warning

    def test_the_exact_reported_query_warns(self) -> None:
        assert any(
            "ORs adjacent terms" in w for w in validate_arxiv_query("sheaf neureal network").warnings
        )

    def test_explicit_operators_do_not_warn(self) -> None:
        for query in [
            "ti:sheaf AND ti:neural",
            "ti:sheaf AND ti:neural AND ti:network",
            "(ti:a OR ti:b) ANDNOT abs:survey",
            "au:del_maestro AND ti:checkerboard",
        ]:
            assert not any(
                "ORs adjacent" in w for w in validate_arxiv_query(query).warnings
            ), query

    def test_field_prefixed_terms_do_not_warn(self) -> None:
        """`cat:cs.CL ti:x` is two clauses, not two loose words."""
        assert not any(
            "ORs adjacent" in w for w in validate_arxiv_query("cat:cs.CL ti:x").warnings
        )

    def test_a_quoted_phrase_does_not_warn(self) -> None:
        assert not any(
            "ORs adjacent" in w for w in validate_arxiv_query('ti:"attention is all you need"').warnings
        )

    def test_a_date_range_does_not_warn(self) -> None:
        query = "submittedDate:[202301010600 TO 202401010600]"
        assert not any("ORs adjacent" in w for w in validate_arxiv_query(query).warnings)

    def test_a_single_bare_word_does_not_warn(self) -> None:
        assert not any("ORs adjacent" in w for w in validate_arxiv_query("sheaf").warnings)

    def test_parentheses_are_not_counted_as_terms(self) -> None:
        assert not any("ORs adjacent" in w for w in validate_arxiv_query("(sheaf)").warnings)

    def test_split_reports_terms_and_operators(self) -> None:
        from app.domain.filters import _split_bare_terms

        assert _split_bare_terms("sheaf neural network") == (["sheaf", "neural", "network"], [])
        assert _split_bare_terms("ti:a AND cat:cs.CL") == ([], ["AND"])
        assert _split_bare_terms('ti:"a b c"') == ([], [])

    def test_the_warning_reaches_every_surface(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        request = search_query_from(raw="sheaf neureal network")
        assert any("ORs adjacent" in w for w in request.warnings)


class TestPhraseSupport:
    """`--phrase` exists because the shell eats quotes.

    `paper search "a b c"` reaches us as `a b c`, which ArXiv then ORs.
    """

    def test_phrase_is_quoted(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        request = search_query_from(phrases=["sheaf neural network"])
        assert request.filter.compile() == '"sheaf neural network"'

    def test_phrases_are_ored(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        request = search_query_from(phrases=["a b", "c d"])
        assert request.filter.compile() == '"a b" OR "c d"'

    def test_already_quoted_phrase_is_left_alone(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        request = search_query_from(phrases=['"a b"'])
        assert request.filter.compile() == '"a b"'

    def test_a_phrase_is_never_flagged_as_bare_terms(self) -> None:
        assert not any(
            "ORs adjacent" in w for w in validate_arxiv_query('"sheaf neural network"').warnings
        )

    def test_raw_is_anded_not_dropped(self) -> None:
        """An unquoted `--phrase a b c` is split by the shell into a phrase plus
        two positional words. Losing the positional part would silently change
        the search, so both survive."""
        from app.clients.arxiv.filters import search_query_from

        request = search_query_from(phrases=["sheaf"], raw="neureal network")
        compiled = request.filter.compile()
        assert '"sheaf"' in compiled
        assert "neureal network" in compiled
        assert " AND " in compiled

    def test_no_leading_operator(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        compiled = search_query_from(phrases=["x"], raw="cat:cs.CL").filter.compile()
        assert not compiled.startswith("AND")
        assert compiled == '"x" AND cat:cs.CL'

    def test_phrase_alone_needs_no_operator(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        assert not search_query_from(phrases=["x y"]).filter.compile().startswith("AND")

    def test_raw_alone_is_unchanged(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        assert search_query_from(raw="ti:x").filter.compile() == "ti:x"

    def test_empty_phrases_fall_through_to_raw(self) -> None:
        from app.clients.arxiv.filters import search_query_from

        assert search_query_from(phrases=["  "], raw="ti:x").filter.compile() == "ti:x"


class TestIdListRequestShape:
    """An id lookup must send `id_list` and nothing else.

    Measured against the live API for the old-style id `cond-mat/0404680v1`:

    =======================================  ======  ======
    request                                  HTTP    entries
    =======================================  ======  ======
    ``search_query=id:cond-mat/0404680v1``    200     0
    ``id_list`` with no ``search_query``     200     1
    both together                             200     0
    =======================================  ======

    All three are HTTP 200. The old-style form is simply not matched by the
    `id:` field, and sending it alongside `id_list` makes ArXiv ignore
    `id_list` — so the failure looked exactly like "this paper does not exist".
    """

    def test_id_list_is_the_only_parameter(self) -> None:
        params = build_search_params(SearchQuery(id_list=("cond-mat/0404680v1",)))
        assert params["id_list"] == "cond-mat/0404680v1"
        assert "search_query" not in params
        # ArXiv answers sortBy alongside id_list with a 500.
        assert "sortBy" not in params
        assert "sortOrder" not in params

    def test_several_ids_are_comma_joined(self) -> None:
        params = build_search_params(
            SearchQuery(id_list=("cond-mat/0404680v1", "1706.03762v7"))
        )
        assert params["id_list"] == "cond-mat/0404680v1,1706.03762v7"

    def test_an_ordinary_query_is_unchanged(self) -> None:
        params = build_search_params(SearchQuery(title_terms=("transformer",)))
        assert params["search_query"] == "ti:transformer"
        assert "id_list" not in params
        assert "sortBy" in params

    def test_id_list_never_reaches_the_search_query_string(self) -> None:
        """`id:` in a search_query silently matches nothing on old-style ids."""
        params = build_search_params(SearchQuery(id_list=("cond-mat/0404680v1",)))
        assert "id:" not in str(params.get("search_query", ""))

