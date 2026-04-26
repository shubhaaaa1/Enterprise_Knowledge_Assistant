"""Bug condition exploration test for Neo4j code presence.

**Validates: Requirements 1.5, 2.5**

This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

This test encodes the expected behavior - it will validate the fix when it passes after implementation.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper functions to scan codebase for Neo4j-related code
# ---------------------------------------------------------------------------


def find_python_files(root_dir: str) -> list[Path]:
    """Find all Python files in the source directory."""
    root = Path(root_dir)
    python_files = []
    
    # Scan src directory
    src_dir = root / "src"
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))
    
    return python_files


def scan_file_for_neo4j_imports(file_path: Path) -> list[int]:
    """Scan a Python file for Neo4j imports.
    
    Returns a list of line numbers where Neo4j imports are found.
    """
    import_lines = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))
        
        # Walk the AST looking for Neo4j imports
        for node in ast.walk(tree):
            # Check for "import neo4j" or "from neo4j import ..."
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "neo4j" in alias.name.lower():
                        import_lines.append(node.lineno)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and "neo4j" in node.module.lower():
                    import_lines.append(node.lineno)
    
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        pass
    
    return import_lines


def scan_file_for_neo4j_classes(file_path: Path) -> dict[str, list[int]]:
    """Scan a Python file for Neo4j-related classes.
    
    Returns a dict mapping class names to line numbers.
    """
    neo4j_classes = {}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))
        
        # Look for GraphStore, EntityNode, RelationshipEdge, DependencyGraph classes
        neo4j_class_names = ["GraphStore", "EntityNode", "RelationshipEdge", "DependencyGraph"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in neo4j_class_names:
                    if node.name not in neo4j_classes:
                        neo4j_classes[node.name] = []
                    neo4j_classes[node.name].append(node.lineno)
    
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        pass
    
    return neo4j_classes


def scan_file_for_neo4j_references(file_path: Path) -> list[int]:
    """Scan a Python file for Neo4j string references.
    
    Returns a list of line numbers where Neo4j references are found.
    """
    reference_lines = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, start=1):
            # Look for Neo4j references in strings, comments, or identifiers
            if "neo4j" in line.lower() or "graphstore" in line.lower():
                # Exclude test files and comments that are just documentation
                if "test_" not in str(file_path):
                    reference_lines.append(line_num)
    
    except (UnicodeDecodeError, IOError):
        # Skip files that can't be read
        pass
    
    return reference_lines


def check_graph_store_file_exists(root_dir: str) -> bool:
    """Check if graph_store.py file exists."""
    graph_store_path = Path(root_dir) / "src" / "enterprise_rag" / "graph_store.py"
    return graph_store_path.exists()


def scan_pyproject_for_neo4j_dependency(root_dir: str) -> list[str]:
    """Scan pyproject.toml for neo4j dependency.
    
    Returns a list of lines containing neo4j dependency.
    """
    pyproject_path = Path(root_dir) / "pyproject.toml"
    neo4j_lines = []
    
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, start=1):
            if "neo4j" in line.lower():
                neo4j_lines.append(f"Line {line_num}: {line.strip()}")
    
    except (FileNotFoundError, UnicodeDecodeError):
        pass
    
    return neo4j_lines


def scan_codebase_for_neo4j(root_dir: str) -> dict:
    """Scan entire codebase for Neo4j-related code.
    
    Returns a dict with:
    - imports: dict mapping file paths to import line numbers
    - classes: dict mapping file paths to class definitions
    - references: dict mapping file paths to reference line numbers
    - graph_store_exists: bool indicating if graph_store.py exists
    - pyproject_dependencies: list of neo4j dependency lines
    """
    results = {
        "imports": {},
        "classes": {},
        "references": {},
        "graph_store_exists": False,
        "pyproject_dependencies": [],
    }
    
    # Check if graph_store.py exists
    results["graph_store_exists"] = check_graph_store_file_exists(root_dir)
    
    # Scan pyproject.toml
    results["pyproject_dependencies"] = scan_pyproject_for_neo4j_dependency(root_dir)
    
    # Scan Python files
    python_files = find_python_files(root_dir)
    
    for file_path in python_files:
        rel_path = str(file_path.relative_to(root_dir))
        
        # Scan for imports
        import_lines = scan_file_for_neo4j_imports(file_path)
        if import_lines:
            results["imports"][rel_path] = import_lines
        
        # Scan for classes
        class_defs = scan_file_for_neo4j_classes(file_path)
        if class_defs:
            results["classes"][rel_path] = class_defs
        
        # Scan for references
        reference_lines = scan_file_for_neo4j_references(file_path)
        if reference_lines:
            results["references"][rel_path] = reference_lines
    
    return results


# ---------------------------------------------------------------------------
# Property 1: Bug Condition - Neo4j Code Still Present in Codebase
# Validates: Requirements 1.5, 2.5
# ---------------------------------------------------------------------------


@given(root_dir=st.just(os.getcwd()))
@settings(max_examples=1, deadline=None)
def test_no_neo4j_code_in_codebase(root_dir):
    """Test that codebase does not contain any Neo4j-related code.
    
    **Validates: Requirements 1.5, 2.5**
    
    Bug Condition 1.5: Neo4j-related code, health checks, and complexity still present throughout the codebase
    
    Expected Behavior 2.5: No Neo4j-related code, imports, models, or health checks should exist
    
    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    DO NOT attempt to fix the test or the code when it fails.
    
    This test encodes the expected behavior - it will validate the fix when it passes after implementation.
    """
    # Scan codebase for Neo4j-related code
    neo4j_usage = scan_codebase_for_neo4j(root_dir)
    
    # Build detailed failure message with counterexamples
    failure_messages = []
    has_neo4j_code = False
    
    # Check for graph_store.py file
    if neo4j_usage["graph_store_exists"]:
        has_neo4j_code = True
        failure_messages.append("\n=== COUNTEREXAMPLE: graph_store.py file exists ===")
        failure_messages.append("  - File: src/enterprise_rag/graph_store.py")
        failure_messages.append("  - This file should be removed (Bug Condition 1.5)")
    
    # Check for Neo4j imports
    if neo4j_usage["imports"]:
        has_neo4j_code = True
        failure_messages.append("\n\n=== COUNTEREXAMPLES: Neo4j imports found ===")
        
        for file_path, line_numbers in sorted(neo4j_usage["imports"].items()):
            failure_messages.append(f"\n{file_path}:")
            failure_messages.append(f"  - Neo4j imports at lines: {line_numbers}")
    
    # Check for Neo4j classes
    if neo4j_usage["classes"]:
        has_neo4j_code = True
        failure_messages.append("\n\n=== COUNTEREXAMPLES: Neo4j-related classes found ===")
        
        for file_path, class_defs in sorted(neo4j_usage["classes"].items()):
            failure_messages.append(f"\n{file_path}:")
            for class_name, line_numbers in class_defs.items():
                failure_messages.append(f"  - Class '{class_name}' at lines: {line_numbers}")
    
    # Check for Neo4j references
    if neo4j_usage["references"]:
        has_neo4j_code = True
        failure_messages.append("\n\n=== COUNTEREXAMPLES: Neo4j references found ===")
        
        # Limit to first 10 files to avoid overwhelming output
        for file_path, line_numbers in sorted(list(neo4j_usage["references"].items())[:10]):
            failure_messages.append(f"\n{file_path}:")
            # Limit to first 20 line numbers per file
            limited_lines = line_numbers[:20]
            if len(line_numbers) > 20:
                failure_messages.append(f"  - Neo4j references at lines: {limited_lines} ... and {len(line_numbers) - 20} more")
            else:
                failure_messages.append(f"  - Neo4j references at lines: {limited_lines}")
        
        if len(neo4j_usage["references"]) > 10:
            failure_messages.append(f"\n  ... and {len(neo4j_usage['references']) - 10} more files with Neo4j references")
    
    # Check for Neo4j dependencies in pyproject.toml
    if neo4j_usage["pyproject_dependencies"]:
        has_neo4j_code = True
        failure_messages.append("\n\n=== COUNTEREXAMPLE: Neo4j dependency in pyproject.toml ===")
        for dep_line in neo4j_usage["pyproject_dependencies"]:
            failure_messages.append(f"  - {dep_line}")
    
    # Summary
    if has_neo4j_code:
        failure_messages.append("\n\n=== SUMMARY ===")
        failure_messages.append(f"  - graph_store.py exists: {neo4j_usage['graph_store_exists']}")
        failure_messages.append(f"  - Files with Neo4j imports: {len(neo4j_usage['imports'])}")
        failure_messages.append(f"  - Files with Neo4j classes: {len(neo4j_usage['classes'])}")
        failure_messages.append(f"  - Files with Neo4j references: {len(neo4j_usage['references'])}")
        failure_messages.append(f"  - Neo4j dependencies in pyproject.toml: {len(neo4j_usage['pyproject_dependencies'])}")
        failure_messages.append("\n  Expected Behavior 2.5: No Neo4j-related code should exist")
    
    # ASSERTION: No Neo4j-related code should be present
    # This will FAIL on unfixed code (which is expected and correct)
    assert not has_neo4j_code, (
        "Neo4j-related code found in codebase. "
        "Expected all Neo4j code, imports, models, and dependencies to be removed."
        + "".join(failure_messages)
    )
