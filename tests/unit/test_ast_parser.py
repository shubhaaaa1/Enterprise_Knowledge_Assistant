"""Unit tests for ASTParser."""

from __future__ import annotations

import textwrap

import pytest

from enterprise_rag.ast_parser import ASTParser
from enterprise_rag.models import CodeSymbol


@pytest.fixture()
def parser() -> ASTParser:
    return ASTParser()


# ---------------------------------------------------------------------------
# supported_languages
# ---------------------------------------------------------------------------

class TestSupportedLanguages:
    def test_python_always_supported(self, parser):
        assert "python" in parser.supported_languages()

    def test_returns_list(self, parser):
        result = parser.supported_languages()
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Non-parseable files
# ---------------------------------------------------------------------------

class TestNonParseableFiles:
    @pytest.mark.parametrize("file_path", [
        "README.md",
        "config.yml",
        "settings.yaml",
        "pyproject.toml",
        "setup.cfg",
        "app.ini",
        "image.png",
        "archive.zip",
    ])
    def test_non_parseable_returns_empty(self, parser, file_path):
        result = parser.parse(file_path, "some content", "unknown")
        assert result == []

    def test_unknown_language_returns_empty(self, parser):
        result = parser.parse("file.xyz", "some content", "cobol")
        assert result == []


# ---------------------------------------------------------------------------
# Python function extraction
# ---------------------------------------------------------------------------

class TestPythonFunctionExtraction:
    def test_simple_function_extracted(self, parser):
        source = textwrap.dedent("""\
            def greet(name):
                \"\"\"Say hello.\"\"\"
                return f"Hello, {name}"
        """)
        symbols = parser.parse("greet.py", source, "python")
        assert len(symbols) == 1
        sym = symbols[0]
        assert sym.symbol_name == "greet"
        assert sym.symbol_type == "function"

    def test_function_line_range(self, parser):
        source = textwrap.dedent("""\
            def foo():
                pass
        """)
        symbols = parser.parse("foo.py", source, "python")
        assert len(symbols) == 1
        assert symbols[0].line_start == 1
        assert symbols[0].line_end == 2

    def test_function_docstring(self, parser):
        source = textwrap.dedent("""\
            def documented():
                \"\"\"This is the docstring.\"\"\"
                pass
        """)
        symbols = parser.parse("doc.py", source, "python")
        assert symbols[0].docstring == "This is the docstring."

    def test_function_without_docstring(self, parser):
        source = textwrap.dedent("""\
            def no_doc():
                pass
        """)
        symbols = parser.parse("nodoc.py", source, "python")
        assert symbols[0].docstring is None

    def test_function_call_refs(self, parser):
        source = textwrap.dedent("""\
            def caller():
                foo()
                bar()
        """)
        symbols = parser.parse("caller.py", source, "python")
        assert len(symbols) == 1
        refs = symbols[0].call_refs
        assert "foo" in refs
        assert "bar" in refs

    def test_function_file_path(self, parser):
        source = "def f(): pass\n"
        symbols = parser.parse("mymodule/utils.py", source, "python")
        assert symbols[0].file_path == "mymodule/utils.py"

    def test_function_symbol_id_is_uuid(self, parser):
        source = "def f(): pass\n"
        symbols = parser.parse("f.py", source, "python")
        import uuid
        # Should not raise
        uuid.UUID(symbols[0].symbol_id)

    def test_multiple_functions(self, parser):
        source = textwrap.dedent("""\
            def alpha():
                pass

            def beta():
                pass

            def gamma():
                pass
        """)
        symbols = parser.parse("multi.py", source, "python")
        names = {s.symbol_name for s in symbols}
        assert names == {"alpha", "beta", "gamma"}
        assert all(s.symbol_type == "function" for s in symbols)

    def test_async_function_extracted(self, parser):
        source = textwrap.dedent("""\
            async def fetch():
                pass
        """)
        symbols = parser.parse("async.py", source, "python")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "fetch"
        assert symbols[0].symbol_type == "function"

    def test_permission_tags_propagated(self, parser):
        source = "def f(): pass\n"
        symbols = parser.parse("f.py", source, "python", permission_tags=["engineering"])
        assert symbols[0].permission_tags == ["engineering"]

    def test_source_id_propagated(self, parser):
        source = "def f(): pass\n"
        symbols = parser.parse("f.py", source, "python", source_id="repo-123")
        assert symbols[0].source_id == "repo-123"

    def test_call_refs_is_list(self, parser):
        source = "def f(): pass\n"
        symbols = parser.parse("f.py", source, "python")
        assert isinstance(symbols[0].call_refs, list)


# ---------------------------------------------------------------------------
# Python class extraction
# ---------------------------------------------------------------------------

class TestPythonClassExtraction:
    def test_class_extracted(self, parser):
        source = textwrap.dedent("""\
            class MyClass:
                \"\"\"A class.\"\"\"
                pass
        """)
        symbols = parser.parse("cls.py", source, "python")
        class_syms = [s for s in symbols if s.symbol_type == "class"]
        assert len(class_syms) == 1
        assert class_syms[0].symbol_name == "MyClass"
        assert class_syms[0].docstring == "A class."

    def test_class_methods_extracted(self, parser):
        source = textwrap.dedent("""\
            class Calculator:
                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b
        """)
        symbols = parser.parse("calc.py", source, "python")
        method_syms = [s for s in symbols if s.symbol_type == "method"]
        method_names = {s.symbol_name for s in method_syms}
        assert "add" in method_names
        assert "subtract" in method_names

    def test_methods_not_duplicated_as_functions(self, parser):
        source = textwrap.dedent("""\
            class Foo:
                def bar(self):
                    pass
        """)
        symbols = parser.parse("foo.py", source, "python")
        func_syms = [s for s in symbols if s.symbol_type == "function"]
        # bar should be a method, not a function
        assert not any(s.symbol_name == "bar" for s in func_syms)

    def test_class_and_top_level_function(self, parser):
        source = textwrap.dedent("""\
            def standalone():
                pass

            class Widget:
                def render(self):
                    pass
        """)
        symbols = parser.parse("mixed.py", source, "python")
        types = {s.symbol_type for s in symbols}
        assert "function" in types
        assert "class" in types
        assert "method" in types

    def test_method_line_range(self, parser):
        source = textwrap.dedent("""\
            class A:
                def method(self):
                    x = 1
                    return x
        """)
        symbols = parser.parse("a.py", source, "python")
        method = next(s for s in symbols if s.symbol_type == "method")
        assert method.line_start == 2
        assert method.line_end == 4

    def test_method_call_refs(self, parser):
        source = textwrap.dedent("""\
            class Service:
                def process(self):
                    helper()
                    self.validate()
        """)
        symbols = parser.parse("svc.py", source, "python")
        method = next(s for s in symbols if s.symbol_name == "process")
        assert "helper" in method.call_refs

    def test_syntax_error_returns_empty(self, parser):
        source = "def broken(:\n    pass\n"
        symbols = parser.parse("broken.py", source, "python")
        assert symbols == []
