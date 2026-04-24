"""Neo4j-backed graph store for the Enterprise RAG System."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from enterprise_rag.models import CodeSymbol, ComponentStatus, DependencyGraph

logger = logging.getLogger(__name__)

# Map symbol_type → Neo4j node label
_LABEL_MAP: Dict[str, str] = {
    "function": "Function",
    "class": "Class",
    "method": "Method",
    "module": "Module",
    "file": "File",
}

_DEFAULT_LABEL = "Function"


def _node_label(symbol_type: str) -> str:
    return _LABEL_MAP.get(symbol_type.lower(), _DEFAULT_LABEL)


class GraphStore:
    """Neo4j graph store for code dependency relationships.

    Uses the bolt protocol via the official ``neo4j`` Python driver.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying driver connection."""
        self._driver.close()

    def upsert_symbols(self, symbols: List[CodeSymbol]) -> None:
        """Create or merge nodes for each CodeSymbol.

        Nodes are merged on (symbol_name, file_path) to avoid duplicates.
        Each node receives the label corresponding to its symbol_type.
        """
        if not symbols:
            return

        with self._driver.session() as session:
            for symbol in symbols:
                label = _node_label(symbol.symbol_type)
                session.run(
                    f"""
                    MERGE (n:{label} {{symbol_name: $symbol_name, file_path: $file_path}})
                    SET n.symbol_id   = $symbol_id,
                        n.symbol_type = $symbol_type,
                        n.docstring   = $docstring,
                        n.line_start  = $line_start,
                        n.line_end    = $line_end,
                        n.source_id   = $source_id
                    """,
                    symbol_name=symbol.symbol_name,
                    file_path=symbol.file_path,
                    symbol_id=symbol.symbol_id,
                    symbol_type=symbol.symbol_type,
                    docstring=symbol.docstring,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                    source_id=symbol.source_id,
                )

    def upsert_relationships(self, symbols: List[CodeSymbol]) -> None:
        """Create edges between nodes that already exist in the graph.

        Only creates edges where both source and target nodes are present
        (no orphaned edges — Req 11.7).  Relationship types created:

        * ``CALLS``       — symbol.call_refs entries
        * ``DEFINED_IN``  — symbol → its file node (if a File node exists)
        * ``CONTAINS``    — file node → symbol (inverse of DEFINED_IN)
        * ``INHERITS``    — not derivable from CodeSymbol alone; skipped here
        * ``IMPORTS``     — not derivable from CodeSymbol alone; skipped here
        """
        if not symbols:
            return

        with self._driver.session() as session:
            for symbol in symbols:
                src_label = _node_label(symbol.symbol_type)

                # CALLS edges — only when target node exists
                for callee_name in symbol.call_refs:
                    session.run(
                        f"""
                        MATCH (src:{src_label} {{symbol_name: $src_name, file_path: $src_file}})
                        MATCH (tgt {{symbol_name: $tgt_name}})
                        MERGE (src)-[:CALLS]->(tgt)
                        """,
                        src_name=symbol.symbol_name,
                        src_file=symbol.file_path,
                        tgt_name=callee_name,
                    )

                # DEFINED_IN / CONTAINS edges — link symbol to its File node
                session.run(
                    f"""
                    MATCH (sym:{src_label} {{symbol_name: $sym_name, file_path: $sym_file}})
                    MATCH (f:File {{file_path: $sym_file}})
                    MERGE (sym)-[:DEFINED_IN]->(f)
                    MERGE (f)-[:CONTAINS]->(sym)
                    """,
                    sym_name=symbol.symbol_name,
                    sym_file=symbol.file_path,
                )

    def query_dependencies(
        self,
        symbol: str,
        direction: Literal["callers", "callees", "both"],
        depth: int = 2,
    ) -> DependencyGraph:
        """Return a subgraph of dependencies for the given symbol.

        Args:
            symbol:    The ``symbol_name`` of the root node.
            direction: ``"callers"`` — nodes that CALL the target;
                       ``"callees"`` — nodes the target CALLS;
                       ``"both"``    — union of callers and callees.
            depth:     Maximum traversal hops (default 2).

        Returns:
            A :class:`DependencyGraph` with ``nodes`` and ``edges`` lists.
        """
        nodes: List[Dict] = []
        edges: List[Dict] = []
        seen_node_ids: set = set()

        with self._driver.session() as session:
            if direction in ("callees", "both"):
                result = session.run(
                    """
                    MATCH path = (root {symbol_name: $symbol})-[:CALLS*1..$depth]->(dep)
                    UNWIND nodes(path) AS n
                    UNWIND relationships(path) AS r
                    RETURN
                        n.symbol_name AS name,
                        n.file_path   AS file_path,
                        labels(n)[0]  AS label,
                        elementId(n)  AS node_id,
                        startNode(r).symbol_name AS src,
                        endNode(r).symbol_name   AS tgt,
                        type(r)                  AS rel_type
                    """,
                    symbol=symbol,
                    depth=depth,
                )
                self._collect_results(result, nodes, edges, seen_node_ids)

            if direction in ("callers", "both"):
                result = session.run(
                    """
                    MATCH path = (caller)-[:CALLS*1..$depth]->(root {symbol_name: $symbol})
                    UNWIND nodes(path) AS n
                    UNWIND relationships(path) AS r
                    RETURN
                        n.symbol_name AS name,
                        n.file_path   AS file_path,
                        labels(n)[0]  AS label,
                        elementId(n)  AS node_id,
                        startNode(r).symbol_name AS src,
                        endNode(r).symbol_name   AS tgt,
                        type(r)                  AS rel_type
                    """,
                    symbol=symbol,
                    depth=depth,
                )
                self._collect_results(result, nodes, edges, seen_node_ids)

        return DependencyGraph(
            nodes=nodes,
            edges=edges,
            root_symbol=symbol,
            direction=direction,
            depth=depth,
        )

    def health_check(self) -> ComponentStatus:
        """Verify connectivity by running a trivial Cypher query."""
        now = datetime.now(tz=timezone.utc)
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").consume()
            return ComponentStatus(
                name="neo4j",
                status="ok",
                last_checked=now,
                detail=f"uri='{self._uri}'",
            )
        except ServiceUnavailable as exc:
            logger.error("Neo4j health check failed: %s", exc)
            return ComponentStatus(
                name="neo4j",
                status="down",
                last_checked=now,
                detail=str(exc),
            )
        except Exception as exc:
            logger.error("Neo4j health check error: %s", exc)
            return ComponentStatus(
                name="neo4j",
                status="degraded",
                last_checked=now,
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_results(
        result,
        nodes: List[Dict],
        edges: List[Dict],
        seen_node_ids: set,
    ) -> None:
        """Accumulate unique nodes and edges from a Cypher result."""
        seen_edges: set = set()

        for record in result:
            node_id = record["node_id"]
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                nodes.append(
                    {
                        "id": node_id,
                        "label": record["label"] or "",
                        "name": record["name"] or "",
                        "file_path": record["file_path"] or "",
                    }
                )

            edge_key = (record["src"], record["tgt"], record["rel_type"])
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": record["src"],
                        "target": record["tgt"],
                        "relationship": record["rel_type"],
                    }
                )
