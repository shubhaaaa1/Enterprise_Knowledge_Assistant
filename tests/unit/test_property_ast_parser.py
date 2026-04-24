"""Property-based tests for ASTParser — AST symbol extraction completeness.

# Feature: enterprise-rag-system, Property 24: AST symbol extraction completeness
"""

from __future__ import annotations

import textwrap

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.ast_parser import ASTParser
from enterprise_rag.models import CodeSymbol

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Valid Python identifiers for names
_identifier = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# A simple function body line (avoids syntax issues)
_body_line = st.just("    pass")


def _function_source(name: str) -> str:
    """Generate a minimal Python function definition."""
    return f"def {name}():\n    pass\n"


def _class_source(class_name: str, method_names: list[str]) -> str:
    """Generate a minimal Python class with methods."""
    lines = [f"class {class_name}:"]
    if method_names:
        for mname in method_names:
            lines.append(f"    def {mname}(self):")
            lines.append("        pass")
    else:
        lines.append("    pass")
    return "\n".join(lines) + "\n"


@st.composite
def python_source_with_symbols(draw) -> str:
    """Generate a Python source snippet containing functions and/or classes."""
    parts = []

    # 0–3 top-level functions
    num_funcs = draw(st.integers(min_value=0, max_value=3))
    func_names = draw(
        st.lists(_identifier, min_size=num_funcs, max_size=num_funcs, unique=True)
    )
    for name in func_names:
        parts.append(_function_source(name))

    # 0–2 classes, each with 0–3 methods
    num_classes = draw(st.integers(min_value=0, max_value=2))
    class_names = draw(
        st.lists(_identifier, min_size=num_classes, max_size=num_classes, unique=True)
    )
    for cname in class_names:
        num_methods = draw(st.integers(min_value=0, max_value=3))
        method_names = draw(
            st.lists(_identifier, min_size=num_methods, max_size=num_methods, unique=True)
        )
        parts.append(_class_source(cname, method_names))

    # Ensure we have at least one definition so the test is meaningful
    if not parts:
        parts.append(_function_source("placeholder"))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Property 24: AST symbol extraction completeness
# Validates: Requirements 1.7
# ---------------------------------------------------------------------------

@given(source=python_source_with_symbols())
@settings(max_examples=20, deadline=None)
def test_ast_symbol_extraction_completeness(source: str) -> None:
    """Every CodeSymbol returned by ASTParser must have non-null symbol_name,
    symbol_type, file_path, line_start, line_end, and call_refs (may be empty
    list but must be present).

    Validates: Requirements 1.7
    """
    parser = ASTParser()
    symbols = parser.parse(
        file_path="generated_test.py",
        content=source,
        language="python",
        source_id="test-source",
        permission_tags=["engineering"],
    )

    for sym in symbols:
        assert isinstance(sym, CodeSymbol), (
            f"Expected CodeSymbol, got {type(sym)}"
        )
        assert sym.symbol_name is not None and sym.symbol_name != "", (
            f"symbol_name must be non-null and non-empty, got {sym.symbol_name!r}"
        )
        assert sym.symbol_type is not None and sym.symbol_type != "", (
            f"symbol_type must be non-null and non-empty, got {sym.symbol_type!r}"
        )
        assert sym.file_path is not None and sym.file_path != "", (
            f"file_path must be non-null and non-empty, got {sym.file_path!r}"
        )
        assert sym.line_start is not None, (
            f"line_start must be non-null for symbol {sym.symbol_name!r}"
        )
        assert sym.line_end is not None, (
            f"line_end must be non-null for symbol {sym.symbol_name!r}"
        )
        assert sym.call_refs is not None, (
            f"call_refs must be present (may be empty list) for symbol {sym.symbol_name!r}"
        )
        assert isinstance(sym.call_refs, list), (
            f"call_refs must be a list, got {type(sym.call_refs)} for {sym.symbol_name!r}"
        )
        assert sym.line_start >= 1, (
            f"line_start must be >= 1, got {sym.line_start} for {sym.symbol_name!r}"
        )
        assert sym.line_end >= sym.line_start, (
            f"line_end ({sym.line_end}) must be >= line_start ({sym.line_start}) "
            f"for symbol {sym.symbol_name!r}"
        )
