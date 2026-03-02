"""Tests for codebase indexing and search."""

import json
import tempfile
from pathlib import Path

import pytest

from forge.tools.codebase import (
    CodebaseIndexer,
    Symbol,
    FileIndex,
    SearchResult,
    DEFAULT_IGNORE,
    CODE_EXTENSIONS,
    MAX_FILE_SIZE,
)


class TestSymbol:
    """Tests for the Symbol dataclass."""

    def test_basic_symbol(self):
        s = Symbol(name="foo", kind="function", file="test.py", line=10)
        assert s.name == "foo"
        assert s.kind == "function"
        assert s.file == "test.py"
        assert s.line == 10

    def test_symbol_with_signature(self):
        s = Symbol(
            name="bar", kind="method", file="cls.py", line=5,
            signature="def bar(self, x: int) -> str",
            parent="MyClass",
        )
        assert "int" in s.signature
        assert s.parent == "MyClass"

    def test_symbol_defaults(self):
        s = Symbol(name="x", kind="variable", file="a.py", line=1)
        assert s.signature == ""
        assert s.docstring == ""
        assert s.parent == ""


class TestFileIndex:
    """Tests for the FileIndex dataclass."""

    def test_basic_file_index(self):
        fi = FileIndex(path="test.py", language="python", size=100, lines=10, hash="abc")
        assert fi.path == "test.py"
        assert fi.language == "python"
        assert fi.symbols == []
        assert fi.imports == []

    def test_file_index_with_symbols(self):
        sym = Symbol(name="func", kind="function", file="test.py", line=1)
        fi = FileIndex(
            path="test.py", language="python", size=100, lines=10,
            hash="abc", symbols=[sym],
        )
        assert len(fi.symbols) == 1
        assert fi.symbols[0].name == "func"


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_basic_result(self):
        r = SearchResult(file="test.py", line=5, content="def foo():")
        assert r.file == "test.py"
        assert r.score == 0.0

    def test_result_with_score(self):
        r = SearchResult(file="test.py", line=5, content="def foo():", score=8.0)
        assert r.score == 8.0


class TestConstants:
    """Tests for module-level constants."""

    def test_default_ignore_has_common_dirs(self):
        assert ".git" in DEFAULT_IGNORE
        assert "__pycache__" in DEFAULT_IGNORE
        assert "node_modules" in DEFAULT_IGNORE
        assert ".venv" in DEFAULT_IGNORE

    def test_code_extensions_has_common_types(self):
        assert ".py" in CODE_EXTENSIONS
        assert ".js" in CODE_EXTENSIONS
        assert ".ts" in CODE_EXTENSIONS
        assert ".go" in CODE_EXTENSIONS
        assert ".rs" in CODE_EXTENSIONS

    def test_max_file_size_reasonable(self):
        assert MAX_FILE_SIZE > 10_000
        assert MAX_FILE_SIZE <= 10_000_000


class TestCodebaseIndexerInit:
    """Tests for CodebaseIndexer initialization."""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer.project_dir == Path(tmpdir).resolve()
            assert indexer.client is None
            assert not indexer._loaded

    def test_custom_ignore_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir, ignore_patterns={"custom_dir"})
            assert "custom_dir" in indexer.ignore_patterns


class TestPythonSymbolExtraction:
    """Tests for Python symbol extraction."""

    def test_extract_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello(name: str) -> str:\n    return name\n")
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_python_symbols(
                "def hello(name: str) -> str:\n    return name\n", "test.py"
            )
            assert len(symbols) >= 1
            assert symbols[0].name == "hello"
            assert symbols[0].kind == "function"
            assert "str" in symbols[0].signature

    def test_extract_class(self):
        code = "class MyClass(Base):\n    pass\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_python_symbols(code, "test.py")
            assert len(symbols) >= 1
            assert symbols[0].name == "MyClass"
            assert symbols[0].kind == "class"
            assert "Base" in symbols[0].signature

    def test_extract_method(self):
        code = "class Foo:\n    def bar(self, x: int) -> bool:\n        pass\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_python_symbols(code, "test.py")
            # Should have class + method
            names = [s.name for s in symbols]
            assert "Foo" in names
            assert "bar" in names
            bar = [s for s in symbols if s.name == "bar"][0]
            assert bar.kind == "method"
            assert bar.parent == "Foo"

    def test_extract_imports(self):
        code = "import os\nfrom pathlib import Path\nimport json\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            imports = indexer._extract_python_imports(code)
            assert "os" in imports
            assert "pathlib" in imports
            assert "json" in imports


class TestJSSymbolExtraction:
    """Tests for JavaScript/TypeScript symbol extraction."""

    def test_extract_function(self):
        code = "function greet(name) {\n  return name;\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_js_symbols(code, "test.js")
            assert len(symbols) >= 1
            assert symbols[0].name == "greet"
            assert symbols[0].kind == "function"

    def test_extract_class(self):
        code = "class MyComponent extends BaseComponent {\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_js_symbols(code, "test.js")
            assert len(symbols) >= 1
            assert symbols[0].name == "MyComponent"
            assert "extends" in symbols[0].signature

    def test_extract_export_function(self):
        code = "export function handleClick(e) {\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_js_symbols(code, "test.js")
            assert len(symbols) >= 1
            assert symbols[0].name == "handleClick"


class TestGoSymbolExtraction:
    """Tests for Go symbol extraction."""

    def test_extract_function(self):
        code = "func main() {\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_go_symbols(code, "main.go")
            assert len(symbols) >= 1
            assert symbols[0].name == "main"

    def test_extract_struct(self):
        code = "type User struct {\n    Name string\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_go_symbols(code, "user.go")
            assert len(symbols) >= 1
            assert symbols[0].name == "User"
            assert symbols[0].kind == "class"


class TestRustSymbolExtraction:
    """Tests for Rust symbol extraction."""

    def test_extract_function(self):
        code = "pub fn process(input: &str) -> Result<String, Error> {\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_rust_symbols(code, "lib.rs")
            assert len(symbols) >= 1
            assert symbols[0].name == "process"

    def test_extract_struct(self):
        code = "pub struct Config {\n    pub name: String,\n}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            symbols = indexer._extract_rust_symbols(code, "lib.rs")
            assert len(symbols) >= 1
            assert symbols[0].name == "Config"
            assert symbols[0].kind == "class"


class TestLanguageDetection:
    """Tests for file extension to language mapping."""

    def test_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer._detect_language(".py") == "python"

    def test_javascript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer._detect_language(".js") == "javascript"
            assert indexer._detect_language(".jsx") == "javascript"

    def test_typescript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer._detect_language(".ts") == "typescript"
            assert indexer._detect_language(".tsx") == "typescript"

    def test_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer._detect_language(".xyz") == "unknown"


class TestBuildIndex:
    """Tests for full index building."""

    def test_empty_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()
            assert stats["files"] == 0
            assert stats["symbols"] == 0

    def test_single_python_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text(
                "def main():\n    print('hello')\n\nclass App:\n    pass\n"
            )
            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()
            assert stats["files"] == 1
            assert stats["symbols"] >= 2  # main + App

    def test_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def func_a(): pass\n")
            (Path(tmpdir) / "b.py").write_text("def func_b(): pass\n")
            (Path(tmpdir) / "c.js").write_text("function funcC() {}\n")
            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()
            assert stats["files"] == 3
            assert stats["symbols"] >= 3

    def test_ignores_pycache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "mod.cpython-310.pyc").write_bytes(b"compiled")
            (Path(tmpdir) / "real.py").write_text("x = 1\n")
            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()
            # Should only index real.py, not the pycache file
            assert stats["files"] == 1

    def test_ignores_non_code_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG")
            (Path(tmpdir) / "data.bin").write_bytes(b"\x00\x01")
            (Path(tmpdir) / "code.py").write_text("x = 1\n")
            indexer = CodebaseIndexer(tmpdir)
            stats = indexer.build_index()
            assert stats["files"] == 1


class TestSearch:
    """Tests for codebase search."""

    def _build_test_index(self, tmpdir):
        (Path(tmpdir) / "auth.py").write_text(
            "class UserAuth:\n"
            "    def login(self, username: str) -> bool:\n"
            "        pass\n"
            "    def logout(self):\n"
            "        pass\n"
        )
        (Path(tmpdir) / "models.py").write_text(
            "class User:\n"
            "    name: str\n"
            "class Product:\n"
            "    price: float\n"
        )
        indexer = CodebaseIndexer(tmpdir)
        indexer.build_index()
        return indexer

    def test_search_by_symbol_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("UserAuth")
            assert len(results) >= 1
            assert any(r.symbol == "UserAuth" for r in results)

    def test_search_by_partial_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("login")
            assert len(results) >= 1

    def test_search_by_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("auth")
            assert len(results) >= 1
            assert any("auth" in r.file for r in results)

    def test_search_max_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("User", max_results=2)
            assert len(results) <= 2

    def test_search_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("zzz_nonexistent_xyz")
            assert len(results) == 0

    def test_search_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = self._build_test_index(tmpdir)
            results = indexer.search("User")
            # Each (file, line) pair should appear at most once
            keys = [(r.file, r.line) for r in results]
            assert len(keys) == len(set(keys))


class TestFindSymbol:
    """Tests for symbol lookup."""

    def test_find_exact_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            syms = indexer.find_symbol("hello")
            assert len(syms) == 1
            assert syms[0].name == "hello"

    def test_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("class MyClass: pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert len(indexer.find_symbol("myclass")) >= 1
            assert len(indexer.find_symbol("MYCLASS")) >= 1

    def test_find_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("x = 1\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert indexer.find_symbol("nonexistent") == []


class TestFileSummary:
    """Tests for file summary retrieval."""

    def test_get_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("def main(): pass\nclass App: pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            summary = indexer.get_file_summary("app.py")
            assert "app.py" in summary
            assert "python" in summary

    def test_file_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            result = indexer.get_file_summary("nonexistent.py")
            assert "not indexed" in result.lower()


class TestProjectOverview:
    """Tests for project overview."""

    def test_empty_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            overview = indexer.get_project_overview()
            assert "No files indexed" in overview

    def test_overview_shows_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("def run(): pass\n")
            src = Path(tmpdir) / "src"
            src.mkdir()
            (src / "app.py").write_text("class App: pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            overview = indexer.get_project_overview()
            assert "main.py" in overview
            assert "Files:" in overview


class TestIndexPersistence:
    """Tests for index save/load round-trip."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello(): pass\n")

            # Build and save
            indexer1 = CodebaseIndexer(tmpdir)
            indexer1.build_index()
            assert len(indexer1._file_index) == 1

            # Load fresh
            indexer2 = CodebaseIndexer(tmpdir)
            indexer2._load_index()
            assert len(indexer2._file_index) == 1
            assert "test.py" in indexer2._file_index
            assert len(indexer2.find_symbol("hello")) == 1

    def test_index_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("x = 1\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert (Path(tmpdir) / ".forge" / "index" / "files.json").exists()

    def test_load_nonexistent_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            indexer._load_index()  # should not crash
            assert indexer._loaded is True
            assert len(indexer._file_index) == 0


class TestIncrementalUpdate:
    """Tests for incremental index update."""

    def test_update_detects_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def func_a(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert indexer._file_index.get("a.py") is not None

            # Add new file
            (Path(tmpdir) / "b.py").write_text("def func_b(): pass\n")
            stats = indexer.update_index()
            assert stats["added"] >= 1
            assert indexer._file_index.get("b.py") is not None

    def test_update_detects_deleted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def func_a(): pass\n")
            (Path(tmpdir) / "b.py").write_text("def func_b(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert len(indexer._file_index) == 2

            # Delete one
            (Path(tmpdir) / "b.py").unlink()
            stats = indexer.update_index()
            assert stats["removed"] >= 1
            assert "b.py" not in indexer._file_index

    def test_update_detects_modified_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def original(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            # Modify file
            (Path(tmpdir) / "a.py").write_text("def modified(): pass\n")
            stats = indexer.update_index()
            assert stats["updated"] >= 1
            # Should have the new symbol
            assert len(indexer.find_symbol("modified")) >= 1

    def test_update_unchanged_file_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def func_a(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()

            stats = indexer.update_index()
            assert stats["updated"] == 0
            assert stats["added"] == 0
            assert stats["removed"] == 0


class TestRemoveFromIndex:
    """Tests for removing files from the index."""

    def test_remove_cleans_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def unique_func(): pass\n")
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            assert len(indexer.find_symbol("unique_func")) == 1

            indexer._remove_from_index("test.py")
            assert len(indexer.find_symbol("unique_func")) == 0
            assert "test.py" not in indexer._file_index

    def test_remove_nonexistent_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            indexer.build_index()
            indexer._remove_from_index("nonexistent.py")  # should not crash


class TestHashFile:
    """Tests for content hash computation."""

    def test_same_content_same_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("x = 1\n")
            (Path(tmpdir) / "b.py").write_text("x = 1\n")
            indexer = CodebaseIndexer(tmpdir)
            ha = indexer._hash_file(Path(tmpdir) / "a.py")
            hb = indexer._hash_file(Path(tmpdir) / "b.py")
            assert ha == hb

    def test_different_content_different_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("x = 1\n")
            (Path(tmpdir) / "b.py").write_text("x = 2\n")
            indexer = CodebaseIndexer(tmpdir)
            ha = indexer._hash_file(Path(tmpdir) / "a.py")
            hb = indexer._hash_file(Path(tmpdir) / "b.py")
            assert ha != hb

    def test_nonexistent_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = CodebaseIndexer(tmpdir)
            assert indexer._hash_file(Path(tmpdir) / "nope.py") == ""
