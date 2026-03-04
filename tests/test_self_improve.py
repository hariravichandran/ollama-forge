"""Tests for the self-improvement agent (no network/git required)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.community.self_improve import (
    ImprovementResult,
    SelfImproveAgent,
    STABLE_PUSH_COOLDOWN,
    UPSTREAM_REPO,
)
from forge.community.ideas import IdeaCollector


class MockClient:
    """Mock OllamaClient for self-improve tests."""

    def __init__(self, responses=None):
        self.model = "test:7b"
        self.base_url = "http://localhost:11434"
        self.num_ctx = 8192
        self.keep_alive = "30m"
        self._responses = responses or []
        self._call_idx = 0

        class Stats:
            total_calls = 0
            total_tokens = 0
            total_prompt_tokens = 0
            errors = 0
            avg_time_s = 0.0

        self.stats = Stats()

    def generate(self, prompt="", system="", temperature=0.7, timeout=300, json_mode=False):
        resp = self._next_response()
        self.stats.total_calls += 1
        return {"response": resp, "tokens": 10}

    def chat(self, messages, tools=None, temperature=0.7, timeout=300):
        resp = self._next_response()
        self.stats.total_calls += 1
        return {"response": resp, "tokens": 10, "prompt_tokens": 5, "time_s": 0.1}

    def _next_response(self):
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        return "{}"


class TestImprovementResult:
    """Tests for ImprovementResult dataclass."""

    def test_basic_fields(self):
        r = ImprovementResult(
            idea_id="abc123",
            success=True,
            description="Added feature X",
            files_changed=["forge/tools/new.py"],
            tests_passed=True,
            committed=True,
        )
        assert r.idea_id == "abc123"
        assert r.success is True
        assert r.tests_passed is True
        assert r.committed is True

    def test_defaults(self):
        r = ImprovementResult(
            idea_id="x", success=False, description="d",
            files_changed=[], tests_passed=False, committed=False,
        )
        assert r.commit_hash == ""
        assert r.pr_url == ""

    def test_with_pr_url(self):
        r = ImprovementResult(
            idea_id="x", success=True, description="d",
            files_changed=["f.py"], tests_passed=True,
            committed=True, pr_url="https://github.com/test/repo/pull/1",
        )
        assert "github.com" in r.pr_url


class TestConstants:
    """Tests for module-level constants."""

    def test_stable_cooldown(self):
        assert STABLE_PUSH_COOLDOWN == 2 * 24 * 3600  # 2 days in seconds

    def test_upstream_repo(self):
        assert UPSTREAM_REPO == "hariravichandran/ollama-forge"


class TestSelfImproveInit:
    """Tests for SelfImproveAgent initialization."""

    def test_default_contributor_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            assert agent.is_contributor is True
            assert agent.maintainer is False

    def test_maintainer_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas,
                repo_dir=tmpdir, maintainer=True,
            )
            assert agent.maintainer is True
            assert agent.is_contributor is False

    def test_state_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            assert agent.state_file.parent.exists()


class TestGatherCandidates:
    """Tests for _gather_candidates."""

    def test_gathers_community_ideas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            ideas.submit("Test idea", "A test improvement")
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            with patch.object(agent, "_search_latest_advances", return_value=[]):
                candidates = agent._gather_candidates()
            assert len(candidates) == 1
            assert candidates[0]["source"] == "community"
            assert candidates[0]["title"] == "Test idea"

    def test_empty_when_no_ideas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            with patch.object(agent, "_search_latest_advances", return_value=[]):
                candidates = agent._gather_candidates()
            assert len(candidates) == 0


class TestEvaluateCandidates:
    """Tests for _evaluate_candidates."""

    def test_returns_none_for_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            result = agent._evaluate_candidates([])
            assert result is None

    def test_returns_candidate_with_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient(responses=[
                json.dumps({"pick": 1, "reason": "High impact", "feasibility": "high"})
            ])
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            candidates = [
                {"title": "Feature A", "description": "desc", "category": "feature", "votes": 3},
                {"title": "Feature B", "description": "desc", "category": "feature", "votes": 1},
            ]
            result = agent._evaluate_candidates(candidates)
            assert result is not None
            assert result["title"] == "Feature A"

    def test_falls_back_to_first_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient(responses=["not valid json at all"])
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            candidates = [{"title": "Only one", "description": "d", "category": "c", "votes": 0}]
            result = agent._evaluate_candidates(candidates)
            assert result is not None
            assert result["title"] == "Only one"


class TestProposeImplementation:
    """Tests for _propose_implementation."""

    def test_returns_dict_on_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient(responses=[
                json.dumps({
                    "files_to_modify": ["forge/cli.py"],
                    "changes": [{"file": "forge/cli.py", "old_code": "x", "new_code": "y"}],
                    "tests_to_run": ["pytest tests/"],
                    "risk": "low",
                })
            ])
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            result = agent._propose_implementation({"title": "Test", "description": "d"})
            assert result is not None
            assert "files_to_modify" in result

    def test_returns_none_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient(responses=["not json"])
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            result = agent._propose_implementation({"title": "Test", "description": "d"})
            assert result is None


class TestRunTests:
    """Tests for _run_tests."""

    def test_passing_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            # Use a safe whitelisted command that succeeds (--version exits 0)
            assert agent._run_tests(["python -m pytest --version"]) is True

    def test_failing_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            # Use a safe whitelisted command that fails (invalid pytest option)
            assert agent._run_tests(["python -m pytest --__invalid_opt_xyz"]) is False

    def test_empty_runs_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            # Default command is "python -m pytest tests/ -x --tb=short"
            # which will fail in a temp dir with no tests — that's fine
            result = agent._run_tests([])
            assert isinstance(result, bool)


class TestStatePersistence:
    """Tests for state save/load."""

    def test_empty_state_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            assert agent.state == {}

    def test_state_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            agent.state["last_stable_push"] = 1000.0
            agent._save_state()

            # Reload
            agent2 = SelfImproveAgent(
                client=MockClient(), idea_collector=ideas, repo_dir=tmpdir,
            )
            assert agent2.state.get("last_stable_push") == 1000.0

    def test_corrupted_state_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".forge_state"
            state_dir.mkdir()
            (state_dir / "self_improve_state.json").write_text("not json{{{")
            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            assert agent.state == {}


class TestFeatureBranch:
    """Tests for _create_feature_branch."""

    def test_branch_name_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize a git repo for git operations
            import subprocess
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmpdir, capture_output=True)
            (Path(tmpdir) / "README.md").write_text("# Test\n")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)

            client = MockClient()
            ideas = IdeaCollector(ideas_dir=str(Path(tmpdir) / "ideas"))
            agent = SelfImproveAgent(
                client=client, idea_collector=ideas, repo_dir=tmpdir,
            )
            branch = agent._create_feature_branch({"title": "Add dark mode support"})
            assert branch.startswith("self-improve/")
            assert "dark-mode" in branch.lower()
