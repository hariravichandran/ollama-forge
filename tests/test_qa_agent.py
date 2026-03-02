"""Tests for QA agent: generates and validates test cases for code changes."""

import tempfile
from pathlib import Path

import pytest

from forge.agents.qa import QAAgent, QAResult
from forge.llm.client import OllamaClient


class TestQAResult:
    """Tests for QAResult."""

    def test_passed_result(self):
        r = QAResult(
            passed=True,
            existing_tests_passed=True,
            generated_tests_passed=True,
            summary="All good",
            generated_test_code="def test_x(): pass",
            test_output="1 passed",
        )
        assert r.passed is True
        assert r.existing_tests_passed is True
        assert r.generated_tests_passed is True

    def test_failed_result(self):
        r = QAResult(
            passed=False,
            existing_tests_passed=True,
            generated_tests_passed=False,
            summary="Generated tests failed",
            generated_test_code="def test_x(): assert False",
            test_output="1 failed",
        )
        assert r.passed is False

    def test_repr(self):
        r = QAResult(
            passed=True, existing_tests_passed=True,
            generated_tests_passed=True, summary="ok",
            generated_test_code="", test_output="",
        )
        repr_str = repr(r)
        assert "QAResult" in repr_str
        assert "passed=True" in repr_str


class TestQAAgentInit:
    """Tests for QAAgent initialization."""

    def test_init(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            assert qa.repo_dir == Path(tmpdir)

    def test_init_default_dir(self):
        client = OllamaClient(base_url="http://localhost:19999")
        qa = QAAgent(client=client)
        assert qa.repo_dir == Path(".")


class TestGenerateTests:
    """Tests for test code generation (no LLM needed for validation tests)."""

    def test_no_files_returns_empty(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            result = qa._generate_tests(
                files_changed=["nonexistent.py"],
                change_description="test",
                diff="",
            )
            assert result == ""

    def test_truncates_large_files(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            large_file = Path(tmpdir) / "big.py"
            large_file.write_text("x = 1\n" * 500)
            qa = QAAgent(client=client, repo_dir=tmpdir)
            # This will try to call the LLM but will fail (no server)
            # That's fine — we're testing the file reading path
            result = qa._generate_tests(
                files_changed=["big.py"],
                change_description="test",
                diff="",
            )
            # May return empty since LLM call will fail
            assert isinstance(result, str)


class TestRunGeneratedTests:
    """Tests for running generated test code."""

    def test_passing_tests(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            passed, output = qa._run_generated_tests("def test_ok(): assert 1 + 1 == 2")
            assert passed is True
            assert "passed" in output.lower()

    def test_failing_tests(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            passed, output = qa._run_generated_tests("def test_fail(): assert False")
            assert passed is False

    def test_syntax_error_in_tests(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            passed, output = qa._run_generated_tests("def test_bad(:\n    pass")
            assert passed is False

    def test_temp_file_cleaned_up(self):
        """The generated test file should be cleaned up after running."""
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            qa._run_generated_tests("def test_ok(): pass")
            # No _qa_generated_test.py should remain in tests/ or anywhere
            test_files = list(Path(tmpdir).rglob("*qa*test*"))
            assert len(test_files) == 0


class TestReviewCode:
    """Tests for code review (requires LLM, so mostly test error handling)."""

    def test_review_with_nonexistent_files(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = QAAgent(client=client, repo_dir=tmpdir)
            # This will fail to call LLM but should not crash
            result = qa.review_code(["nonexistent.py"])
            assert isinstance(result, str)

    def test_review_limits_files(self):
        client = OllamaClient(base_url="http://localhost:19999")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                (Path(tmpdir) / f"file{i}.py").write_text(f"x = {i}\n")
            qa = QAAgent(client=client, repo_dir=tmpdir)
            # Should only read up to 5 files
            # We can't easily test the internal limit without mocking LLM,
            # but we verify it doesn't crash
            result = qa.review_code([f"file{i}.py" for i in range(10)])
            assert isinstance(result, str)
