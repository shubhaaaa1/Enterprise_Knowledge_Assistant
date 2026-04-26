"""Bug condition exploration test for deprecated datetime methods.

**Validates: Requirements 1.1, 1.2**

This test MUST FAIL on unfixed code - failure confirms the bugs exist.
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
# Helper functions to scan codebase for deprecated datetime patterns
# ---------------------------------------------------------------------------


def find_python_files(root_dir: str) -> list[Path]:
    """Find all Python files in the source directory."""
    root = Path(root_dir)
    python_files = []
    
    # Only scan src directory, not tests
    src_dir = root / "src"
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))
    
    return python_files


def scan_file_for_deprecated_datetime(file_path: Path) -> dict[str, list[int]]:
    """Scan a Python file for deprecated datetime methods.
    
    Returns a dict with keys 'utcnow' and 'utcfromtimestamp', each containing
    a list of line numbers where the deprecated method is found.
    """
    results = {"utcnow": [], "utcfromtimestamp": []}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))
        
        # Walk the AST looking for deprecated datetime calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for datetime.utcnow()
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "utcnow":
                        # Check if it's called on datetime
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "datetime":
                            results["utcnow"].append(node.lineno)
                    
                    # Check for datetime.utcfromtimestamp()
                    elif node.func.attr == "utcfromtimestamp":
                        # Check if it's called on datetime
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "datetime":
                            results["utcfromtimestamp"].append(node.lineno)
    
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        pass
    
    return results


def scan_codebase_for_deprecated_datetime(root_dir: str) -> dict[str, dict[str, list[int]]]:
    """Scan entire codebase for deprecated datetime methods.
    
    Returns a dict mapping file paths to their deprecated datetime usage.
    """
    all_results = {}
    
    python_files = find_python_files(root_dir)
    
    for file_path in python_files:
        results = scan_file_for_deprecated_datetime(file_path)
        
        # Only include files that have deprecated usage
        if results["utcnow"] or results["utcfromtimestamp"]:
            # Store relative path for cleaner output
            rel_path = str(file_path.relative_to(root_dir))
            all_results[rel_path] = results
    
    return all_results


def check_timezone_aware_replacements(root_dir: str) -> dict[str, dict[str, list[int]]]:
    """Check if timezone-aware datetime methods are used instead of deprecated ones.
    
    Returns a dict mapping file paths to their timezone-aware datetime usage.
    """
    all_results = {}
    
    python_files = find_python_files(root_dir)
    
    for file_path in python_files:
        results = {"now_with_utc": [], "fromtimestamp_with_tz": []}
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Walk the AST looking for timezone-aware datetime calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check for datetime.now(timezone.utc) or datetime.now(tz=timezone.utc)
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr == "now":
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == "datetime":
                                # Check if timezone.utc is passed as argument
                                has_tz_arg = False
                                for arg in node.args:
                                    if isinstance(arg, ast.Attribute) and arg.attr == "utc":
                                        has_tz_arg = True
                                        break
                                for keyword in node.keywords:
                                    if keyword.arg in ("tz", None):
                                        if isinstance(keyword.value, ast.Attribute) and keyword.value.attr == "utc":
                                            has_tz_arg = True
                                            break
                                
                                if has_tz_arg:
                                    results["now_with_utc"].append(node.lineno)
                        
                        # Check for datetime.fromtimestamp(ts, tz=timezone.utc)
                        elif node.func.attr == "fromtimestamp":
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == "datetime":
                                # Check if tz=timezone.utc is passed
                                has_tz_kwarg = False
                                for keyword in node.keywords:
                                    if keyword.arg == "tz":
                                        if isinstance(keyword.value, ast.Attribute) and keyword.value.attr == "utc":
                                            has_tz_kwarg = True
                                            break
                                
                                if has_tz_kwarg:
                                    results["fromtimestamp_with_tz"].append(node.lineno)
        
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed
            pass
        
        # Only include files that have timezone-aware usage
        if results["now_with_utc"] or results["fromtimestamp_with_tz"]:
            rel_path = str(file_path.relative_to(root_dir))
            all_results[rel_path] = results
    
    return all_results


# ---------------------------------------------------------------------------
# Property 1: Bug Condition - Deprecated datetime.utcnow() and utcfromtimestamp() Usage
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------


@given(root_dir=st.just(os.getcwd()))
@settings(max_examples=1, deadline=None)
def test_deprecated_datetime_methods_not_used(root_dir):
    """Test that codebase uses timezone-aware datetime methods instead of deprecated ones.
    
    **Validates: Requirements 1.1, 1.2**
    
    Bug Condition 1.1: System uses deprecated datetime.utcnow() method in 8+ locations
    Bug Condition 1.2: System uses deprecated datetime.utcfromtimestamp() method in 2 locations in DocsConnector
    
    Expected Behavior 2.1: Use timezone-aware datetime.now(timezone.utc) instead
    Expected Behavior 2.2: Use datetime.fromtimestamp(ts, tz=timezone.utc) instead
    
    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bugs exist.
    DO NOT attempt to fix the test or the code when it fails.
    
    This test encodes the expected behavior - it will validate the fix when it passes after implementation.
    """
    # Scan codebase for deprecated datetime usage
    deprecated_usage = scan_codebase_for_deprecated_datetime(root_dir)
    
    # Scan codebase for timezone-aware replacements
    timezone_aware_usage = check_timezone_aware_replacements(root_dir)
    
    # Count total deprecated usages
    total_utcnow = sum(len(results["utcnow"]) for results in deprecated_usage.values())
    total_utcfromtimestamp = sum(len(results["utcfromtimestamp"]) for results in deprecated_usage.values())
    
    # Count DocsConnector specific usage
    docs_connector_utcfromtimestamp = 0
    for file_path, results in deprecated_usage.items():
        if "docs_connector.py" in file_path:
            docs_connector_utcfromtimestamp += len(results["utcfromtimestamp"])
    
    # Build detailed failure message with counterexamples
    failure_messages = []
    
    if deprecated_usage:
        failure_messages.append("\n=== COUNTEREXAMPLES: Deprecated datetime methods found ===\n")
        
        for file_path, results in sorted(deprecated_usage.items()):
            if results["utcnow"]:
                failure_messages.append(f"\n{file_path}:")
                failure_messages.append(f"  - datetime.utcnow() found at lines: {results['utcnow']}")
            
            if results["utcfromtimestamp"]:
                failure_messages.append(f"\n{file_path}:")
                failure_messages.append(f"  - datetime.utcfromtimestamp() found at lines: {results['utcfromtimestamp']}")
        
        failure_messages.append(f"\n\nTotal deprecated datetime.utcnow() calls: {total_utcnow}")
        failure_messages.append(f"Total deprecated datetime.utcfromtimestamp() calls: {total_utcfromtimestamp}")
        failure_messages.append(f"DocsConnector datetime.utcfromtimestamp() calls: {docs_connector_utcfromtimestamp}")
    
    if timezone_aware_usage:
        failure_messages.append("\n\n=== Timezone-aware datetime methods found (expected after fix) ===\n")
        
        for file_path, results in sorted(timezone_aware_usage.items()):
            if results["now_with_utc"]:
                failure_messages.append(f"\n{file_path}:")
                failure_messages.append(f"  - datetime.now(timezone.utc) found at lines: {results['now_with_utc']}")
            
            if results["fromtimestamp_with_tz"]:
                failure_messages.append(f"\n{file_path}:")
                failure_messages.append(f"  - datetime.fromtimestamp(ts, tz=timezone.utc) found at lines: {results['fromtimestamp_with_tz']}")
    
    # ASSERTION: No deprecated datetime methods should be used
    # This will FAIL on unfixed code (which is expected and correct)
    assert not deprecated_usage, (
        "Deprecated datetime methods found in codebase. "
        "Expected timezone-aware replacements instead."
        + "".join(failure_messages)
    )
    
    # ASSERTION: Timezone-aware methods should be used
    # This will also FAIL on unfixed code (which is expected and correct)
    assert timezone_aware_usage, (
        "No timezone-aware datetime methods found. "
        "Expected datetime.now(timezone.utc) and datetime.fromtimestamp(ts, tz=timezone.utc) to be used."
    )
