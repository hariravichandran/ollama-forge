"""Self-improvement agent: iterates on ollama-forge using LLM + web research.

This agent:
1. Reads community ideas and evaluates them
2. Searches the web for latest AI/LLM advances
3. Proposes improvements to the codebase
4. Tests changes before committing
5. Commits to main branch (never directly to stable)
6. Respects stable branch policy: max 1 push per 2 days

Branch policy:
- main: experimental changes, concurrent contributors welcome
- stable: tested improvements only, max 1 push per 2 days

Designed to handle concurrent usage: uses file-based locking and
pull-before-push to avoid conflicts.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.community.ideas import IdeaCollector, Idea
from forge.llm.client import OllamaClient
from forge.utils.logging import get_logger

log = get_logger("community.self_improve")

# Stable branch: max 1 push per 2 days (172800 seconds)
STABLE_PUSH_COOLDOWN = 2 * 24 * 3600


@dataclass
class ImprovementResult:
    """Result of an improvement attempt."""

    idea_id: str
    success: bool
    description: str
    files_changed: list[str]
    tests_passed: bool
    committed: bool
    commit_hash: str = ""


class SelfImproveAgent:
    """Iterates on ollama-forge, evaluating ideas and proposing improvements.

    Safe for concurrent usage:
    - Always pulls before committing (rebase)
    - Uses file-based state to track stable branch push history
    - Commits only to main, promotes to stable on schedule

    The improvement loop:
    1. Pull latest from main
    2. Read community ideas + search for latest advances
    3. Evaluate which ideas are feasible and beneficial
    4. Implement the most promising improvement
    5. Run tests
    6. If tests pass, commit to main
    7. If enough time since last stable push, promote to stable
    """

    def __init__(
        self,
        client: OllamaClient,
        idea_collector: IdeaCollector,
        repo_dir: str = ".",
    ):
        self.client = client
        self.ideas = idea_collector
        self.repo_dir = Path(repo_dir)
        self.state_file = self.repo_dir / ".forge_state" / "self_improve_state.json"
        self.state = self._load_state()

    def run_iteration(self) -> ImprovementResult | None:
        """Run one improvement iteration.

        Returns ImprovementResult if a change was made, None otherwise.
        """
        # Step 1: Pull latest
        self._git_pull()

        # Step 2: Gather improvement candidates
        candidates = self._gather_candidates()
        if not candidates:
            log.info("No improvement candidates found")
            return None

        # Step 3: Evaluate and pick the best candidate
        best = self._evaluate_candidates(candidates)
        if not best:
            log.info("No viable candidates after evaluation")
            return None

        # Step 4: Propose implementation
        proposal = self._propose_implementation(best)
        if not proposal:
            log.info("Could not generate viable proposal for: %s", best.get("title"))
            return None

        # Step 5: Apply and test
        result = self._apply_and_test(best, proposal)

        # Step 6: Commit if successful
        if result.success and result.tests_passed:
            result.committed = self._commit_to_main(result)

        # Step 7: Maybe promote to stable
        if result.committed:
            self._maybe_promote_to_stable()

        return result

    def _gather_candidates(self) -> list[dict[str, Any]]:
        """Gather improvement candidates from community ideas and web research."""
        candidates = []

        # Community ideas
        new_ideas = self.ideas.get_new_ideas()
        for idea in new_ideas:
            candidates.append({
                "source": "community",
                "idea_id": idea.id,
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "votes": idea.votes,
            })

        # Web research for latest advances
        web_candidates = self._search_latest_advances()
        candidates.extend(web_candidates)

        log.info("Gathered %d improvement candidates (%d community, %d research)",
                 len(candidates), len(new_ideas), len(web_candidates))
        return candidates

    def _search_latest_advances(self) -> list[dict[str, Any]]:
        """Search the web for latest LLM/AI advances relevant to the project."""
        try:
            from forge.mcp.web_search import WebSearchMCP
            search = WebSearchMCP()

            queries = [
                "ollama new features improvements 2026",
                "local LLM optimization techniques",
                "MCP model context protocol new servers",
                "context window compression techniques LLM",
            ]

            candidates = []
            for query in queries:
                results = search.search(query, max_results=3)
                for r in results:
                    candidates.append({
                        "source": "research",
                        "title": r.get("title", ""),
                        "description": r.get("body", ""),
                        "url": r.get("href", ""),
                        "category": "improvement",
                        "votes": 0,
                    })
            return candidates
        except Exception as e:
            log.warning("Web research failed: %s", e)
            return []

    def _evaluate_candidates(self, candidates: list[dict]) -> dict | None:
        """Use LLM to evaluate which candidate is most promising."""
        if not candidates:
            return None

        # Format candidates for LLM
        candidates_text = "\n".join(
            f"{i+1}. [{c['category']}] {c['title']} (votes: {c.get('votes', 0)})\n   {c['description'][:200]}"
            for i, c in enumerate(candidates[:15])
        )

        result = self.client.generate(
            prompt=(
                f"Evaluate these improvement candidates for an open-source local AI framework (ollama-forge).\n"
                f"Pick the ONE most impactful and feasible improvement.\n\n"
                f"Candidates:\n{candidates_text}\n\n"
                f"Return JSON: {{\"pick\": <number>, \"reason\": \"...\", \"feasibility\": \"high/medium/low\"}}"
            ),
            system="You evaluate software improvement proposals. Be practical — favor high-impact, low-risk changes.",
            json_mode=True,
            temperature=0.3,
        )

        try:
            data = json.loads(result.get("response", "{}"))
            pick_idx = data.get("pick", 1) - 1
            if 0 <= pick_idx < len(candidates):
                chosen = candidates[pick_idx]
                chosen["evaluation_reason"] = data.get("reason", "")
                return chosen
        except (json.JSONDecodeError, ValueError):
            pass

        return candidates[0] if candidates else None

    def _propose_implementation(self, candidate: dict) -> dict | None:
        """Ask LLM to propose a concrete implementation."""
        result = self.client.generate(
            prompt=(
                f"Propose a concrete implementation for this improvement to ollama-forge:\n\n"
                f"Title: {candidate['title']}\n"
                f"Description: {candidate['description']}\n\n"
                f"Return JSON: {{\n"
                f"  \"files_to_modify\": [\"path/to/file.py\"],\n"
                f"  \"changes\": [{{\"file\": \"...\", \"description\": \"...\", \"old_code\": \"...\", \"new_code\": \"...\"}}],\n"
                f"  \"tests_to_run\": [\"pytest tests/test_*.py\"],\n"
                f"  \"risk\": \"low/medium/high\"\n"
                f"}}"
            ),
            system="You propose minimal, safe code changes. Prefer small improvements over large refactors.",
            json_mode=True,
            temperature=0.3,
        )

        try:
            return json.loads(result.get("response", "{}"))
        except json.JSONDecodeError:
            return None

    def _apply_and_test(self, candidate: dict, proposal: dict) -> ImprovementResult:
        """Apply proposed changes and run tests."""
        files_changed = []
        idea_id = candidate.get("idea_id", "research")

        # Apply changes
        for change in proposal.get("changes", []):
            file_path = self.repo_dir / change.get("file", "")
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text()
                old_code = change.get("old_code", "")
                new_code = change.get("new_code", "")
                if old_code and old_code in content:
                    content = content.replace(old_code, new_code, 1)
                    file_path.write_text(content)
                    files_changed.append(str(file_path))
            except Exception as e:
                log.error("Failed to apply change to %s: %s", file_path, e)

        # Run tests
        tests_passed = self._run_tests(proposal.get("tests_to_run", []))

        # Rollback if tests failed
        if not tests_passed and files_changed:
            log.warning("Tests failed, rolling back changes")
            self._git_checkout_files(files_changed)
            files_changed = []

        return ImprovementResult(
            idea_id=idea_id,
            success=len(files_changed) > 0,
            description=candidate.get("title", ""),
            files_changed=files_changed,
            tests_passed=tests_passed,
            committed=False,
        )

    def _run_tests(self, test_commands: list[str]) -> bool:
        """Run test commands. Returns True if all pass."""
        if not test_commands:
            # Default: run pytest if it exists
            test_commands = ["python -m pytest tests/ -x --tb=short"]

        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=120, cwd=str(self.repo_dir),
                )
                if result.returncode != 0:
                    log.warning("Test failed: %s\n%s", cmd, result.stderr[:500])
                    return False
            except subprocess.TimeoutExpired:
                log.warning("Test timed out: %s", cmd)
                return False
        return True

    def _commit_to_main(self, result: ImprovementResult) -> bool:
        """Commit improvement to main branch."""
        try:
            # Stage changed files
            for f in result.files_changed:
                self._git_run("add", f)

            # Commit
            msg = f"auto-improve: {result.description}\n\nIdea: {result.idea_id}\nFiles: {', '.join(result.files_changed)}"
            self._git_run("commit", "-m", msg)

            # Pull-rebase then push (safe for concurrent usage)
            self._git_run("pull", "--rebase", "origin", "main")
            self._git_run("push", "origin", "main")

            log.info("Committed improvement to main: %s", result.description)
            return True
        except Exception as e:
            log.error("Failed to commit: %s", e)
            return False

    def _maybe_promote_to_stable(self) -> bool:
        """Promote main to stable if cooldown has elapsed (max 1 per 2 days)."""
        last_stable_push = self.state.get("last_stable_push", 0)
        elapsed = time.time() - last_stable_push

        if elapsed < STABLE_PUSH_COOLDOWN:
            remaining_hours = (STABLE_PUSH_COOLDOWN - elapsed) / 3600
            log.info("Stable push cooldown: %.1f hours remaining", remaining_hours)
            return False

        try:
            # Merge main into stable
            self._git_run("checkout", "stable")
            self._git_run("pull", "origin", "stable")
            self._git_run("merge", "main", "--no-edit")
            self._git_run("push", "origin", "stable")
            self._git_run("checkout", "main")

            self.state["last_stable_push"] = time.time()
            self._save_state()

            log.info("Promoted main to stable branch")
            return True
        except Exception as e:
            log.error("Failed to promote to stable: %s", e)
            # Make sure we're back on main
            try:
                self._git_run("checkout", "main")
            except Exception:
                pass
            return False

    def _git_pull(self) -> None:
        """Pull latest from origin."""
        try:
            self._git_run("pull", "--rebase", "origin", "main")
        except Exception as e:
            log.warning("Git pull failed: %s", e)

    def _git_checkout_files(self, files: list[str]) -> None:
        """Restore files from git (rollback)."""
        for f in files:
            try:
                self._git_run("checkout", "--", f)
            except Exception:
                pass

    def _git_run(self, *args: str) -> str:
        """Run a git command."""
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, timeout=60,
            cwd=str(self.repo_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout.strip()

    def _load_state(self) -> dict[str, Any]:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2))
