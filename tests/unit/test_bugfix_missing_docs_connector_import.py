"""Bug condition exploration test for missing logger import in api.py.

**Validates: Requirements 1.3**

This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

This test encodes the expected behavior - it will validate the fix when it passes after implementation.

NOTE: During investigation, we discovered the actual bug is not DocsConnector import (which is present),
but rather a missing 'logger' import/definition that causes NameError when DOCS_PATH is configured.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper functions to check logger usage and import in api.py
# ---------------------------------------------------------------------------


def check_logger_usage_in_api(root_dir: str) -> dict[str, any]:
    """Check if logger is used but not imported/defined in api.py.
    
    Returns a dict with:
    - 'logger_used': bool indicating if logger is referenced
    - 'logger_usage_lines': list of line numbers where logger is used
    - 'logger_imported': bool indicating if logger is imported
    - 'logger_defined': bool indicating if logger is defined as a variable
    - 'structured_logger_imported': bool indicating if StructuredLogger is imported
    """
    result = {
        'logger_used': False,
        'logger_usage_lines': [],
        'logger_imported': False,
        'logger_defined': False,
        'structured_logger_imported': False,
    }
    
    api_path = Path(root_dir) / "src" / "enterprise_rag" / "api.py"
    
    if not api_path.exists():
        return result
    
    try:
        with open(api_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split('\n')
        
        # Scan for logger usage in source text
        for i, line in enumerate(lines, start=1):
            if "logger." in line and "# " not in line[:line.find("logger.") if "logger." in line else 0]:
                result['logger_used'] = True
                result['logger_usage_lines'].append(i)
        
        # Parse the AST to check for imports and definitions
        tree = ast.parse(content, filename=str(api_path))
        
        for node in ast.walk(tree):
            # Check for logger import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logger" or alias.name == "logging":
                        result['logger_imported'] = True
            
            if isinstance(node, ast.ImportFrom):
                if node.module and "logging" in node.module:
                    for alias in node.names:
                        if alias.name == "logger":
                            result['logger_imported'] = True
                
                # Check for StructuredLogger import
                if node.module == "enterprise_rag.logging":
                    for alias in node.names:
                        if alias.name == "StructuredLogger":
                            result['structured_logger_imported'] = True
            
            # Check for logger variable assignment
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "logger":
                        result['logger_defined'] = True
    
    except (SyntaxError, UnicodeDecodeError):
        pass
    
    return result


def check_docs_connector_context(root_dir: str) -> dict[str, any]:
    """Check the context around DocsConnector usage in api.py.
    
    Returns a dict with:
    - 'docs_connector_line': line number where DocsConnector is instantiated
    - 'logger_call_after_docs': bool indicating if logger is called after DocsConnector
    - 'in_same_function': bool indicating if they're in the same function
    """
    result = {
        'docs_connector_line': None,
        'logger_call_after_docs': False,
        'in_same_function': False,
    }
    
    api_path = Path(root_dir) / "src" / "enterprise_rag" / "api.py"
    
    if not api_path.exists():
        return result
    
    try:
        with open(api_path, "r", encoding="utf-8") as f:
            lines = content = f.read().split('\n')
        
        docs_connector_line = None
        logger_lines = []
        
        for i, line in enumerate(lines, start=1):
            if "DocsConnector(" in line and "import" not in line:
                docs_connector_line = i
                result['docs_connector_line'] = i
            
            if "logger.info" in line:
                logger_lines.append(i)
        
        # Check if logger is called after DocsConnector in the same function
        if docs_connector_line:
            for logger_line in logger_lines:
                if logger_line > docs_connector_line and logger_line - docs_connector_line < 10:
                    result['logger_call_after_docs'] = True
                    result['in_same_function'] = True
    
    except (SyntaxError, UnicodeDecodeError):
        pass
    
    return result


# ---------------------------------------------------------------------------
# Property 1: Bug Condition - Missing logger in api.py causes NameError
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------


@given(root_dir=st.just(os.getcwd()))
@settings(max_examples=1, deadline=None)
def test_logger_available_in_api(root_dir):
    """Test that logger is properly imported or defined in api.py.
    
    **Validates: Requirements 1.3**
    
    Bug Condition 1.3: When DOCS_PATH is configured, api.py crashes with 
                       NameError because logger is not imported/defined
    
    Expected Behavior 2.3: logger should be successfully imported or defined
                          so logging calls work without runtime errors
    
    CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
    DO NOT attempt to fix the test or the code when it fails.
    
    This test encodes the expected behavior - it will validate the fix when it passes after implementation.
    
    NOTE: Investigation revealed the actual bug is missing 'logger', not DocsConnector.
    When DOCS_PATH is set, _make_ingestion_pipeline() calls logger.info() but logger is undefined.
    """
    # Check logger usage and import in api.py
    logger_check = check_logger_usage_in_api(root_dir)
    
    # Check DocsConnector context
    docs_context = check_docs_connector_context(root_dir)
    
    # Build detailed failure message with counterexamples
    failure_messages = []
    
    failure_messages.append("\n=== Logger Usage Analysis ===\n")
    failure_messages.append(f"logger is used: {logger_check['logger_used']}")
    failure_messages.append(f"logger usage lines: {logger_check['logger_usage_lines']}")
    failure_messages.append(f"logger is imported: {logger_check['logger_imported']}")
    failure_messages.append(f"logger is defined: {logger_check['logger_defined']}")
    failure_messages.append(f"StructuredLogger is imported: {logger_check['structured_logger_imported']}")
    
    failure_messages.append("\n=== DocsConnector Context ===\n")
    failure_messages.append(f"DocsConnector instantiation line: {docs_context['docs_connector_line']}")
    failure_messages.append(f"logger called after DocsConnector: {docs_context['logger_call_after_docs']}")
    failure_messages.append(f"In same function: {docs_context['in_same_function']}")
    
    if logger_check['logger_used'] and not logger_check['logger_imported'] and not logger_check['logger_defined']:
        failure_messages.append("\n=== COUNTEREXAMPLE: Missing logger ===\n")
        failure_messages.append("logger is used but not imported or defined in api.py")
        failure_messages.append(f"logger is used at lines: {logger_check['logger_usage_lines']}")
        failure_messages.append("This will cause NameError when DOCS_PATH is configured")
        failure_messages.append("NameError: name 'logger' is not defined")
    
    # ASSERTION 1: If logger is used, it must be imported or defined
    # This will FAIL on unfixed code (which is expected and correct)
    if logger_check['logger_used']:
        assert logger_check['logger_imported'] or logger_check['logger_defined'], (
            "logger is used but not imported or defined in api.py. "
            "This will cause NameError when the code executes."
            + "".join(failure_messages)
        )
    
    # ASSERTION 2: Verify the specific context - when DOCS_PATH is set, logger is called
    # This confirms the bug manifests when DocsConnector is configured
    if docs_context['logger_call_after_docs']:
        assert logger_check['logger_imported'] or logger_check['logger_defined'], (
            "logger.info() is called after DocsConnector instantiation, "
            "but logger is not defined. This causes NameError when DOCS_PATH is configured."
            + "".join(failure_messages)
        )
