"""AST Parser for extracting structured CodeSymbols from source files."""

from __future__ import annotations

import ast
import logging
import uuid
from typing import List, Optional

from enterprise_rag.models import CodeSymbol

logger = logging.getLogger(__name__)

# File extensions that are non-parseable — fall back to text chunking
_NON_PARSEABLE_EXTENSIONS = {
    ".md", ".markdown",
    ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".conf", ".env",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bin", ".exe", ".so", ".dylib",
    ".lock",
}

# Mapping from language name to file extensions
_LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "java": {".java"},
    "go": {".go"},
    "rust": {".rs"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".h"},
    "c": {".c", ".h"},
    "ruby": {".rb"},
    "php": {".php"},
}


def _ext(file_path: str) -> str:
    """Return the lowercased file extension including the dot."""
    dot = file_path.rfind(".")
    if dot == -1:
        return ""
    return file_path[dot:].lower()


# ---------------------------------------------------------------------------
# Python AST extraction helpers
# ---------------------------------------------------------------------------

def _get_call_refs(node: ast.AST) -> List[str]:
    """Walk an AST node and collect all called function/method names."""
    refs: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                refs.append(func.id)
            elif isinstance(func, ast.Attribute):
                refs.append(func.attr)
    return refs


def _extract_source(content: str, line_start: int, line_end: int) -> str:
    """Extract source lines (1-indexed, inclusive)."""
    lines = content.splitlines()
    return "\n".join(lines[line_start - 1 : line_end])


def _parse_python(
    file_path: str,
    content: str,
    source_id: str,
    permission_tags: List[str],
) -> List[CodeSymbol]:
    """Parse a Python source file and return a list of CodeSymbols."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as exc:
        logger.warning("Failed to parse Python file %s: %s", file_path, exc)
        return []

    symbols: List[CodeSymbol] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_symbol = CodeSymbol(
                symbol_id=str(uuid.uuid4()),
                file_path=file_path,
                symbol_name=node.name,
                symbol_type="class",
                docstring=ast.get_docstring(node),
                source_code=_extract_source(content, node.lineno, node.end_lineno),
                line_start=node.lineno,
                line_end=node.end_lineno,
                call_refs=_get_call_refs(node),
                source_id=source_id,
                permission_tags=list(permission_tags),
            )
            symbols.append(class_symbol)

            # Extract methods inside the class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_symbol = CodeSymbol(
                        symbol_id=str(uuid.uuid4()),
                        file_path=file_path,
                        symbol_name=item.name,
                        symbol_type="method",
                        docstring=ast.get_docstring(item),
                        source_code=_extract_source(content, item.lineno, item.end_lineno),
                        line_start=item.lineno,
                        line_end=item.end_lineno,
                        call_refs=_get_call_refs(item),
                        source_id=source_id,
                        permission_tags=list(permission_tags),
                    )
                    symbols.append(method_symbol)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions (not methods — those are handled above)
            # Check if parent is a class by inspecting the tree
            # We use a simple approach: collect all class-level function names
            pass

    # Second pass: collect top-level functions (not inside any class)
    class_method_ids: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_method_ids.add(id(item))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if id(node) not in class_method_ids:
                func_symbol = CodeSymbol(
                    symbol_id=str(uuid.uuid4()),
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    docstring=ast.get_docstring(node),
                    source_code=_extract_source(content, node.lineno, node.end_lineno),
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    call_refs=_get_call_refs(node),
                    source_id=source_id,
                    permission_tags=list(permission_tags),
                )
                symbols.append(func_symbol)

    return symbols


# ---------------------------------------------------------------------------
# Tree-sitter extraction helpers
# ---------------------------------------------------------------------------

def _parse_with_tree_sitter(
    file_path: str,
    content: str,
    language: str,
    source_id: str,
    permission_tags: List[str],
) -> List[CodeSymbol]:
    """Parse a non-Python file using tree-sitter. Returns [] if unavailable."""
    try:
        import tree_sitter  # noqa: F401
    except ImportError:
        logger.warning(
            "tree-sitter is not installed; cannot parse %s (%s). "
            "Falling back to text chunking.",
            file_path,
            language,
        )
        return []

    # Attempt to load the language grammar
    try:
        from tree_sitter import Language, Parser as TSParser

        # Try to load the language — this requires tree-sitter language bindings
        # (e.g. tree-sitter-python, tree-sitter-javascript, etc.)
        lang_module_name = f"tree_sitter_{language.replace('-', '_')}"
        try:
            import importlib
            lang_module = importlib.import_module(lang_module_name)
            ts_language = Language(lang_module.language())
        except (ImportError, AttributeError, Exception) as exc:
            logger.warning(
                "tree-sitter language grammar for '%s' not available (%s). "
                "Falling back to text chunking.",
                language,
                exc,
            )
            return []

        parser = TSParser(ts_language)
        tree = parser.parse(content.encode("utf-8"))
        symbols: List[CodeSymbol] = []

        # Generic extraction: find function/class/method nodes
        _walk_tree_sitter(
            tree.root_node,
            content,
            file_path,
            language,
            source_id,
            permission_tags,
            symbols,
            parent_type=None,
        )
        return symbols

    except Exception as exc:
        logger.warning(
            "tree-sitter parsing failed for %s: %s. Falling back to text chunking.",
            file_path,
            exc,
        )
        return []


def _walk_tree_sitter(
    node,
    content: str,
    file_path: str,
    language: str,
    source_id: str,
    permission_tags: List[str],
    symbols: List[CodeSymbol],
    parent_type: Optional[str],
) -> None:
    """Recursively walk a tree-sitter node tree and extract symbols."""
    function_node_types = {
        "function_definition", "function_declaration",
        "method_definition", "arrow_function",
        "async_function_declaration",
    }
    class_node_types = {
        "class_definition", "class_declaration",
    }

    node_type = node.type

    if node_type in class_node_types or node_type in function_node_types:
        # Determine symbol type
        if node_type in class_node_types:
            symbol_type = "class"
        elif parent_type in class_node_types:
            symbol_type = "method"
        else:
            symbol_type = "function"

        # Extract name
        name_node = node.child_by_field_name("name")
        symbol_name = name_node.text.decode("utf-8") if name_node else "<anonymous>"

        line_start = node.start_point[0] + 1  # tree-sitter is 0-indexed
        line_end = node.end_point[0] + 1

        source_code = content[node.start_byte : node.end_byte]

        symbols.append(
            CodeSymbol(
                symbol_id=str(uuid.uuid4()),
                file_path=file_path,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                docstring=None,  # tree-sitter docstring extraction is language-specific
                source_code=source_code,
                line_start=line_start,
                line_end=line_end,
                call_refs=[],  # call ref extraction via tree-sitter is complex; left empty
                source_id=source_id,
                permission_tags=list(permission_tags),
            )
        )

    for child in node.children:
        _walk_tree_sitter(
            child,
            content,
            file_path,
            language,
            source_id,
            permission_tags,
            symbols,
            parent_type=node_type,
        )


# ---------------------------------------------------------------------------
# ASTParser public class
# ---------------------------------------------------------------------------

class ASTParser:
    """Parses source code files and extracts structured CodeSymbols.

    For Python files the built-in ``ast`` module is used.
    For other supported languages, ``tree-sitter`` bindings are used.
    Non-parseable files (markdown, YAML, config, binary) return an empty list
    so the caller can fall back to standard text chunking.
    """

    def parse(
        self,
        file_path: str,
        content: str,
        language: str,
        source_id: str = "",
        permission_tags: Optional[List[str]] = None,
    ) -> List[CodeSymbol]:
        """Parse *content* and return a list of :class:`CodeSymbol` objects.

        Parameters
        ----------
        file_path:
            The path of the file being parsed (used for metadata and to infer
            the file type when *language* is ambiguous).
        content:
            The raw text content of the file.
        language:
            The programming language of the file (e.g. ``"python"``,
            ``"javascript"``).  Case-insensitive.
        source_id:
            The identifier of the ingestion source (propagated to each symbol).
        permission_tags:
            RBAC tags propagated to each symbol.

        Returns
        -------
        List[CodeSymbol]
            An empty list is returned for non-parseable files or when parsing
            fails; the caller should fall back to text chunking in that case.
        """
        if permission_tags is None:
            permission_tags = []

        ext = _ext(file_path)

        # Non-parseable by extension
        if ext in _NON_PARSEABLE_EXTENSIONS:
            return []

        lang = language.lower().strip()

        if lang == "python" or ext == ".py":
            return _parse_python(file_path, content, source_id, permission_tags)

        # For other languages, attempt tree-sitter
        if lang in _LANGUAGE_EXTENSIONS or ext in {
            e for exts in _LANGUAGE_EXTENSIONS.values() for e in exts
        }:
            return _parse_with_tree_sitter(
                file_path, content, lang, source_id, permission_tags
            )

        # Unknown / unsupported language — return empty list
        logger.debug(
            "Language '%s' (file: %s) is not supported; returning empty list.",
            language,
            file_path,
        )
        return []

    def supported_languages(self) -> List[str]:
        """Return the list of languages this parser can handle.

        Python is always supported.  Other languages are listed if
        ``tree-sitter`` is importable (the actual grammar packages may or may
        not be installed).
        """
        langs = ["python"]
        try:
            import tree_sitter  # noqa: F401
            langs.extend(
                [k for k in _LANGUAGE_EXTENSIONS if k != "python"]
            )
        except ImportError:
            pass
        return langs
