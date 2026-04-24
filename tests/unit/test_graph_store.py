"""Unit tests for GraphStore — all Neo4j I/O is mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from enterprise_rag.graph_store import GraphStore, _node_label
from enterprise_rag.models import CodeSymbol, ComponentStatus, DependencyGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol(
    name: str,
    symbol_type: str = "function",
    file_path: str = "src/foo.py",
    call_refs: List[str] | None = None,
) -> CodeSymbol:
    return CodeSymbol(
        symbol_id=f"id-{name}",
        file_path=file_path,
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

        store = GraphStore(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        return store, mock_driver


# ---------------------------------------------------------------------------
# _node_label helper
# ---------------------------------------------------------------------------

class TestNodeLabel:
    def test_known_types(self):
        assert _node_label("function") == "Function"
        assert _node_label("class") == "Class"
        assert _node_label("method") == "Method"
        assert _node_label("module") == "Module"
        assert _node_label("file") == "File"

    def test_case_insensitive(self):
        assert _node_label("FUNCTION") == "Function"
        assert _node_label("Class") == "Class"

    def test_unknown_defaults_to_function(self):
        assert _node_label("unknown_type") == "Function"


# ---------------------------------------------------------------------------
# upsert_symbols
# ---------------------------------------------------------------------------

class TestUpsertSymbols:
    def test_empty_list_does_not_open_session(self):
        store, mock_driver = _make_graph_store()
        store.upsert_symbols([])
        mock_driver.session.assert_not_called()

    def test_single_symbol_issues_merge_query(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        sym = _make_symbol("my_func", symbol_type="function")
        store.upsert_symbols([sym])

        assert mock_session.run.call_count == 1
        cypher, kwargs = mock_session.run.call_args[0][0], mock_session.run.call_args[1]
        assert "MERGE" in cypher
        assert "Function" in cypher
        assert kwargs["symbol_name"] == "my_func"
        assert kwargs["file_path"] == "src/foo.py"
        assert kwargs["symbol_id"] == "id-my_func"

    def test_multiple_symbols_each_get_a_merge(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        symbols = [
            _make_symbol("func_a", "function"),
            _make_symbol("MyClass", "class"),
            _make_symbol("my_method", "method"),
        ]
        store.upsert_symbols(symbols)

        assert mock_session.run.call_count == 3

    def test_class_symbol_uses_class_label(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store.upsert_symbols([_make_symbol("MyClass", "class")])

        cypher = mock_session.run.call_args[0][0]
        assert "Class" in cypher

    def test_module_symbol_uses_module_label(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store.upsert_symbols([_make_symbol("my_module", "module")])

        cypher = mock_session.run.call_args[0][0]
        assert "Module" in cypher


# ---------------------------------------------------------------------------
# upsert_relationships
# ---------------------------------------------------------------------------

class TestUpsertRelationships:
    def test_empty_list_does_not_open_session(self):
        store, mock_driver = _make_graph_store()
        store.upsert_relationships([])
        mock_driver.session.assert_not_called()

    def test_calls_edge_issued_for_each_call_ref(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        sym = _make_symbol("caller_func", call_refs=["callee_a", "callee_b"])
        store.upsert_relationships([sym])

        # 2 CALLS queries + 1 DEFINED_IN/CONTAINS query
        assert mock_session.run.call_count == 3

        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        calls_cyphers = [c for c in all_cyphers if "CALLS" in c]
        assert len(calls_cyphers) == 2

    def test_calls_edge_uses_match_not_merge_for_nodes(self):
        """Edges must only be created when both nodes exist (no orphaned edges)."""
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        sym = _make_symbol("func_x", call_refs=["func_y"])
        store.upsert_relationships([sym])

        calls_cypher = next(
            c[0][0] for c in mock_session.run.call_args_list if "CALLS" in c[0][0]
        )
        # Both source and target must be MATCHed (not MERGEd) to prevent orphans
        assert calls_cypher.count("MATCH") >= 2

    def test_defined_in_and_contains_edges_issued(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        sym = _make_symbol("my_func", call_refs=[])
        store.upsert_relationships([sym])

        all_cyphers = [c[0][0] for c in mock_session.run.call_args_list]
        structural_cyphers = [c for c in all_cyphers if "DEFINED_IN" in c or "CONTAINS" in c]
        assert len(structural_cyphers) == 1

    def test_no_call_refs_only_structural_edges(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        sym = _make_symbol("isolated_func", call_refs=[])
        store.upsert_relationships([sym])

        # Only the DEFINED_IN/CONTAINS query, no CALLS queries
        assert mock_session.run.call_count == 1


# ---------------------------------------------------------------------------
# query_dependencies
# ---------------------------------------------------------------------------

def _make_record(
    name: str,
    file_path: str,
    label: str,
    node_id: str,
    src: str,
    tgt: str,
    rel_type: str = "CALLS",
) -> MagicMock:
    record = MagicMock()
    record.__getitem__ = lambda self, key: {
        "name": name,
        "file_path": file_path,
        "label": label,
        "node_id": node_id,
        "src": src,
        "tgt": tgt,
        "rel_type": rel_type,
    }[key]
    return record


class TestQueryDependencies:
    def _setup_session(self, mock_driver, records_by_call: list[list]):
        """Configure mock session to return successive record lists per run() call."""
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        side_effects = []
        for records in records_by_call:
            mock_result = MagicMock()
            mock_result.__iter__ = MagicMock(return_value=iter(records))
            side_effects.append(mock_result)

        mock_session.run.side_effect = side_effects
        return mock_session

    def test_callees_direction_issues_outbound_query(self):
        store, mock_driver = _make_graph_store()
        self._setup_session(mock_driver, [[]])

        store.query_dependencies("my_func", direction="callees", depth=2)

        mock_session = mock_driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 1
        cypher = mock_session.run.call_args[0][0]
        # Outbound: root -[:CALLS]-> dep
        assert "root" in cypher and "dep" in cypher

    def test_callers_direction_issues_inbound_query(self):
        store, mock_driver = _make_graph_store()
        self._setup_session(mock_driver, [[]])

        store.query_dependencies("my_func", direction="callers", depth=2)

        mock_session = mock_driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 1
        cypher = mock_session.run.call_args[0][0]
        # Inbound: caller -[:CALLS]-> root
        assert "caller" in cypher and "root" in cypher

    def test_both_direction_issues_two_queries(self):
        store, mock_driver = _make_graph_store()
        self._setup_session(mock_driver, [[], []])

        store.query_dependencies("my_func", direction="both", depth=2)

        mock_session = mock_driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 2

    def test_returns_dependency_graph_dataclass(self):
        store, mock_driver = _make_graph_store()
        self._setup_session(mock_driver, [[]])

        result = store.query_dependencies("my_func", direction="callees")

        assert isinstance(result, DependencyGraph)
        assert result.root_symbol == "my_func"
        assert result.direction == "callees"
        assert result.depth == 2

    def test_depth_passed_to_cypher(self):
        store, mock_driver = _make_graph_store()
        self._setup_session(mock_driver, [[]])

        store.query_dependencies("my_func", direction="callees", depth=5)

        mock_session = mock_driver.session.return_value.__enter__.return_value
        kwargs = mock_session.run.call_args[1]
        assert kwargs["depth"] == 5

    def test_nodes_and_edges_populated_from_records(self):
        store, mock_driver = _make_graph_store()

        rec = _make_record(
            name="callee_func",
            file_path="src/bar.py",
            label="Function",
            node_id="node-1",
            src="my_func",
            tgt="callee_func",
        )
        self._setup_session(mock_driver, [[rec]])

        result = store.query_dependencies("my_func", direction="callees")

        assert len(result.nodes) == 1
        assert result.nodes[0]["name"] == "callee_func"
        assert len(result.edges) == 1
        assert result.edges[0]["source"] == "my_func"
        assert result.edges[0]["target"] == "callee_func"
        assert result.edges[0]["relationship"] == "CALLS"

    def test_duplicate_nodes_deduplicated(self):
        store, mock_driver = _make_graph_store()

        # Same node_id appears twice (e.g. from two path expansions)
        rec1 = _make_record("func_a", "src/a.py", "Function", "node-1", "root", "func_a")
        rec2 = _make_record("func_a", "src/a.py", "Function", "node-1", "root", "func_a")
        self._setup_session(mock_driver, [[rec1, rec2]])

        result = store.query_dependencies("root", direction="callees")

        assert len(result.nodes) == 1


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_ok_when_query_succeeds(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_session.run.return_value = mock_result

        status = store.health_check()

        assert isinstance(status, ComponentStatus)
        assert status.name == "neo4j"
        assert status.status == "ok"
        assert status.last_checked is not None

        mock_session.run.assert_called_once_with("RETURN 1")

    def test_returns_down_on_service_unavailable(self):
        from neo4j.exceptions import ServiceUnavailable

        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = ServiceUnavailable("connection refused")

        status = store.health_check()

        assert status.name == "neo4j"
        assert status.status == "down"
        assert "connection refused" in (status.detail or "")

    def test_returns_degraded_on_unexpected_exception(self):
        store, mock_driver = _make_graph_store()

        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = RuntimeError("unexpected error")

        status = store.health_check()

        assert status.name == "neo4j"
        assert status.status == "degraded"
