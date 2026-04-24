"""Property-based tests for GraphStore — graph edge validity and query direction correctness."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.graph_store import GraphStore
from enterprise_rag.models import CodeSymbol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOL_TYPES = ["function", "class", "method", "module"]


def _make_symbol(name: str, symbol_type: str = "function", call_refs: List[str] | None = None) -> CodeSymbol:
    return CodeSymbol(
        symbol_id=f"id-{name}",
        file_path="src/test.py",
        symbol_name=name,
        symbol_type=symbol_type,
        docstring=None,
        source_code="",
        line_start=1,
        line_end=10,
        call_refs=call_refs or [],
        source_id="repo-1",
        permission_tags=["engineering"],
    )


def _make_graph_store() -> tuple[GraphStore, MagicMock]:
    """Return a GraphStore whose driver is fully mocked."""
    with patch("enterprise_rag.graph_store.GraphDatabase.driver") as mock_driver_cls:
        mock_driver = MagicMock()
        mock_driver_cls.return_value = mock_driver
        store = GraphStore(uri="bolt://localhost:7687", username="neo4j", password="password")
        return store, mock_driver


def _setup_session(mock_driver: MagicMock) -> MagicMock:
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_session


# ---------------------------------------------------------------------------
# Property 26: No orphaned CALLS edges
# Feature: enterprise-rag-system, Property 26: No orphaned CALLS edges
# Validates: Requirements 11.2, 11.7
# ---------------------------------------------------------------------------

_symbol_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

_code_symbol_st = st.builds(
    _make_symbol,
    name=_symbol_name_st,
    symbol_type=st.sampled_from(_SYMBOL_TYPES),
    call_refs=st.lists(_symbol_name_st, max_size=5),
)


@given(symbols=st.lists(_code_symbol_st, min_size=0, max_size=10))
@settings(max_examples=20, deadline=None)
def test_property_26_calls_edges_use_match_not_merge(symbols: List[CodeSymbol]):
    """Every CALLS edge Cypher query must use MATCH (not MERGE) for both endpoints.

    This ensures upsert_relationships never creates orphaned edges — it only
    creates a CALLS edge when both source and target nodes already exist.

    Validates: Requirements 11.2, 11.7
    """
    # Feature: enterprise-rag-system, Property 26: No orphaned CALLS edges
    store, mock_driver = _make_graph_store()
    mock_session = _setup_session(mock_driver)

    store.upsert_relationships(symbols)

    # Collect all Cypher strings issued for CALLS edges
    calls_cyphers = [
        call_args[0][0]
        for call_args in mock_session.run.call_args_list
        if "CALLS" in call_args[0][0]
    ]

    for cypher in calls_cyphers:
        # Both source and target must be MATCHed — not MERGEd — to prevent orphans
        match_count = cypher.upper().count("MATCH")
        assert match_count >= 2, (
            f"CALLS edge Cypher must MATCH both endpoints (found {match_count} MATCH), "
            f"but got:\n{cypher}"
        )
        # The edge itself may use MERGE (for the relationship), but node lookups must be MATCH.
        # A node-binding MERGE looks like: MERGE (n:Label {prop: $val})
        # An edge MERGE looks like: MERGE (a)-[:REL]->(b)  — this is acceptable.
        import re
        node_merge_pattern = re.compile(r"MERGE\s*\(\w+\s*:", re.IGNORECASE)
        node_merges = node_merge_pattern.findall(cypher)
        assert not node_merges, (
            f"CALLS edge Cypher must not MERGE node endpoints (only MATCH), "
            f"found node-binding MERGEs: {node_merges}\nCypher:\n{cypher}"
        )


@given(symbols=st.lists(_code_symbol_st, min_size=1, max_size=10))
@settings(max_examples=20, deadline=None)
def test_property_26_calls_edge_count_matches_call_refs(symbols: List[CodeSymbol]):
    """The number of CALLS Cypher queries equals the total number of call_refs across all symbols.

    Validates: Requirements 11.2, 11.7
    """
    # Feature: enterprise-rag-system, Property 26: No orphaned CALLS edges
    store, mock_driver = _make_graph_store()
    mock_session = _setup_session(mock_driver)

    store.upsert_relationships(symbols)

    total_call_refs = sum(len(s.call_refs) for s in symbols)
    calls_cyphers = [
        call_args[0][0]
        for call_args in mock_session.run.call_args_list
        if "CALLS" in call_args[0][0]
    ]

    assert len(calls_cyphers) == total_call_refs, (
        f"Expected {total_call_refs} CALLS queries (one per call_ref), "
        f"got {len(calls_cyphers)}"
    )


# ---------------------------------------------------------------------------
# Property 27: Dependency query direction correctness
# Feature: enterprise-rag-system, Property 27: Dependency query direction correctness
# Validates: Requirements 11.4
# ---------------------------------------------------------------------------

_direction_st = st.sampled_from(["callers", "callees", "both"])
_symbol_name_nonempty_st = st.text(min_size=1, max_size=30)


def _setup_session_with_empty_results(mock_driver: MagicMock, num_calls: int) -> MagicMock:
    """Configure mock session to return empty results for each run() call."""
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    side_effects = []
    for _ in range(num_calls):
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        side_effects.append(mock_result)

    mock_session.run.side_effect = side_effects
    return mock_session


@given(
    symbol=_symbol_name_nonempty_st,
    direction=_direction_st,
)
@settings(max_examples=20, deadline=None)
def test_property_27_query_direction_issues_correct_cypher_pattern(symbol: str, direction: str):
    """query_dependencies issues Cypher with the correct arrow direction per mode.

    - callees: outbound pattern  root -[:CALLS*]-> dep
    - callers: inbound pattern   caller -[:CALLS*]-> root
    - both:    two queries, one for each direction

    Validates: Requirements 11.4
    """
    # Feature: enterprise-rag-system, Property 27: Dependency query direction correctness
    expected_query_count = 2 if direction == "both" else 1
    store, mock_driver = _make_graph_store()
    _setup_session_with_empty_results(mock_driver, expected_query_count)

    store.query_dependencies(symbol, direction=direction, depth=2)

    mock_session = mock_driver.session.return_value.__enter__.return_value
    assert mock_session.run.call_count == expected_query_count, (
        f"direction='{direction}' should issue {expected_query_count} query/queries, "
        f"got {mock_session.run.call_count}"
    )

    cyphers = [call_args[0][0] for call_args in mock_session.run.call_args_list]

    if direction == "callees":
        # Outbound: root node followed by -[:CALLS*]-> dep
        cypher = cyphers[0]
        assert "root" in cypher, f"callees query must reference 'root' node, got:\n{cypher}"
        assert "dep" in cypher, f"callees query must reference 'dep' node, got:\n{cypher}"
        # Arrow must be outbound: root -> dep (root appears before dep in the path)
        root_pos = cypher.index("root")
        dep_pos = cypher.index("dep")
        assert root_pos < dep_pos, (
            f"callees query must have root before dep (outbound), got:\n{cypher}"
        )

    elif direction == "callers":
        # Inbound: caller node followed by -[:CALLS*]-> root
        cypher = cyphers[0]
        assert "caller" in cypher, f"callers query must reference 'caller' node, got:\n{cypher}"
        assert "root" in cypher, f"callers query must reference 'root' node, got:\n{cypher}"
        # Arrow must be inbound: caller -> root (caller appears before root in the path)
        caller_pos = cypher.index("caller")
        root_pos = cypher.index("root")
        assert caller_pos < root_pos, (
            f"callers query must have caller before root (inbound), got:\n{cypher}"
        )

    elif direction == "both":
        # Must have one outbound (callees) and one inbound (callers) query
        has_outbound = any("root" in c and "dep" in c for c in cyphers)
        has_inbound = any("caller" in c and "root" in c for c in cyphers)
        assert has_outbound, f"'both' must include an outbound (callees) query, got:\n{cyphers}"
        assert has_inbound, f"'both' must include an inbound (callers) query, got:\n{cyphers}"


@given(
    symbol=_symbol_name_nonempty_st,
    direction=_direction_st,
    depth=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=20, deadline=None)
def test_property_27_depth_parameter_passed_to_all_queries(symbol: str, direction: str, depth: int):
    """The depth parameter is forwarded to every Cypher query issued.

    Validates: Requirements 11.4
    """
    # Feature: enterprise-rag-system, Property 27: Dependency query direction correctness
    expected_query_count = 2 if direction == "both" else 1
    store, mock_driver = _make_graph_store()
    _setup_session_with_empty_results(mock_driver, expected_query_count)

    store.query_dependencies(symbol, direction=direction, depth=depth)

    mock_session = mock_driver.session.return_value.__enter__.return_value
    for call_args in mock_session.run.call_args_list:
        kwargs = call_args[1]
        assert kwargs.get("depth") == depth, (
            f"Expected depth={depth} in query kwargs, got {kwargs}"
        )
