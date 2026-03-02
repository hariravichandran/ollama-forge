"""Tests for the multi-file edit planner."""

import tempfile
from pathlib import Path

import pytest

from forge.agents.planner import (
    EditPlanner,
    EditPlan,
    FileEdit,
    PlanResult,
)


class TestFileEdit:
    """Tests for FileEdit dataclass."""

    def test_basic_edit(self):
        fe = FileEdit(path="test.py", description="Fix bug")
        assert fe.path == "test.py"
        assert fe.edits == []
        assert fe.create is False

    def test_edit_with_changes(self):
        fe = FileEdit(
            path="test.py",
            description="Rename variable",
            edits=[{"old_string": "foo", "new_string": "bar"}],
        )
        assert len(fe.edits) == 1

    def test_create_file_edit(self):
        fe = FileEdit(
            path="new.py",
            description="New module",
            create=True,
            new_content="# new file\n",
        )
        assert fe.create is True
        assert fe.new_content == "# new file\n"


class TestEditPlan:
    """Tests for EditPlan dataclass."""

    def test_empty_plan(self):
        plan = EditPlan(task="Do something")
        assert plan.file_count == 0
        assert plan.task == "Do something"

    def test_plan_with_files(self):
        plan = EditPlan(
            task="Rename class",
            files=[
                FileEdit(path="a.py", description="Rename in a"),
                FileEdit(path="b.py", description="Rename in b"),
            ],
        )
        assert plan.file_count == 2

    def test_summary(self):
        plan = EditPlan(
            task="Fix import",
            reasoning="Import path changed",
            files=[
                FileEdit(
                    path="main.py",
                    description="Update import",
                    edits=[{"old_string": "old_mod", "new_string": "new_mod"}],
                ),
            ],
        )
        summary = plan.summary()
        assert "Fix import" in summary
        assert "main.py" in summary
        assert "1 edit" in summary

    def test_summary_create_file(self):
        plan = EditPlan(
            task="Add module",
            files=[FileEdit(path="new.py", description="New", create=True)],
        )
        summary = plan.summary()
        assert "CREATE" in summary


class TestPlanResult:
    """Tests for PlanResult dataclass."""

    def test_success_result(self):
        r = PlanResult(success=True, files_modified=["a.py", "b.py"])
        assert r.success is True
        assert len(r.files_modified) == 2
        assert r.rolled_back is False

    def test_failure_result(self):
        r = PlanResult(success=False, errors=["File not found"], rolled_back=True)
        assert r.success is False
        assert r.rolled_back is True


class TestEditPlannerInit:
    """Tests for EditPlanner initialization."""

    def test_init_without_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            assert planner.client is None

    def test_plan_without_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = planner.plan("Do something")
            assert "No LLM" in plan.reasoning


class TestExecutePlan:
    """Tests for plan execution."""

    def test_execute_single_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("x = 1\ny = 2\n")
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Change x",
                files=[FileEdit(
                    path="test.py",
                    description="Change x to 42",
                    edits=[{"old_string": "x = 1", "new_string": "x = 42"}],
                )],
            )
            result = planner.execute(plan)
            assert result.success
            assert "test.py" in result.files_modified
            assert (Path(tmpdir) / "test.py").read_text() == "x = 42\ny = 2\n"

    def test_execute_multiple_edits_same_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("a = 1\nb = 2\nc = 3\n")
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Change values",
                files=[FileEdit(
                    path="test.py",
                    description="Update a and c",
                    edits=[
                        {"old_string": "a = 1", "new_string": "a = 10"},
                        {"old_string": "c = 3", "new_string": "c = 30"},
                    ],
                )],
            )
            result = planner.execute(plan)
            assert result.success
            content = (Path(tmpdir) / "test.py").read_text()
            assert "a = 10" in content
            assert "c = 30" in content

    def test_execute_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("import old_mod\n")
            (Path(tmpdir) / "b.py").write_text("from old_mod import func\n")
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Rename module",
                files=[
                    FileEdit(
                        path="a.py", description="Update import",
                        edits=[{"old_string": "import old_mod", "new_string": "import new_mod"}],
                    ),
                    FileEdit(
                        path="b.py", description="Update from import",
                        edits=[{"old_string": "from old_mod", "new_string": "from new_mod"}],
                    ),
                ],
                dependency_order=["a.py", "b.py"],
            )
            result = planner.execute(plan)
            assert result.success
            assert len(result.files_modified) == 2

    def test_execute_create_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Add file",
                files=[FileEdit(
                    path="new_module.py",
                    description="Create new module",
                    create=True,
                    new_content="# New module\ndef hello(): pass\n",
                )],
            )
            result = planner.execute(plan)
            assert result.success
            assert (Path(tmpdir) / "new_module.py").exists()
            assert "hello" in (Path(tmpdir) / "new_module.py").read_text()

    def test_execute_create_nested_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Add nested file",
                files=[FileEdit(
                    path="sub/dir/module.py",
                    description="Create nested",
                    create=True,
                    new_content="x = 1\n",
                )],
            )
            result = planner.execute(plan)
            assert result.success
            assert (Path(tmpdir) / "sub" / "dir" / "module.py").exists()


class TestRollback:
    """Tests for atomic rollback on failure."""

    def test_rollback_on_missing_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("original_a\n")
            (Path(tmpdir) / "b.py").write_text("original_b\n")
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Failing edit",
                files=[
                    FileEdit(
                        path="a.py", description="Edit a",
                        edits=[{"old_string": "original_a", "new_string": "modified_a"}],
                    ),
                    FileEdit(
                        path="b.py", description="Edit b (will fail)",
                        edits=[{"old_string": "nonexistent_string", "new_string": "new"}],
                    ),
                ],
                dependency_order=["a.py", "b.py"],
            )
            result = planner.execute(plan)
            assert not result.success
            assert result.rolled_back
            # a.py should be restored to original
            assert (Path(tmpdir) / "a.py").read_text() == "original_a\n"

    def test_rollback_on_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = EditPlanner(working_dir=tmpdir)
            plan = EditPlan(
                task="Edit missing file",
                files=[FileEdit(
                    path="nonexistent.py",
                    description="This will fail",
                    edits=[{"old_string": "x", "new_string": "y"}],
                )],
            )
            result = planner.execute(plan)
            assert not result.success
            assert len(result.errors) > 0


class TestDependencyAnalysis:
    """Tests for Python import dependency analysis."""

    def test_no_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("x = 1\n")
            (Path(tmpdir) / "b.py").write_text("y = 2\n")
            planner = EditPlanner(working_dir=tmpdir)
            deps = planner._analyze_dependencies(["a.py", "b.py"])
            assert deps.get("a.py") == []
            assert deps.get("b.py") == []

    def test_import_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "utils.py").write_text("def helper(): pass\n")
            (Path(tmpdir) / "main.py").write_text("import utils\n")
            planner = EditPlanner(working_dir=tmpdir)
            deps = planner._analyze_dependencies(["utils.py", "main.py"])
            assert "utils.py" in deps.get("main.py", [])

    def test_from_import_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "helpers.py").write_text("def func(): pass\n")
            (Path(tmpdir) / "app.py").write_text("from helpers import func\n")
            planner = EditPlanner(working_dir=tmpdir)
            deps = planner._analyze_dependencies(["helpers.py", "app.py"])
            assert "helpers.py" in deps.get("app.py", [])


class TestGetProjectFiles:
    """Tests for project file discovery."""

    def test_discovers_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("# main")
            (Path(tmpdir) / "readme.md").write_text("# readme")
            planner = EditPlanner(working_dir=tmpdir)
            files = planner._get_project_files()
            assert "main.py" in files
            # .md not in the project file extensions
            assert "readme.md" not in files

    def test_ignores_pycache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "__pycache__"
            cache.mkdir()
            (cache / "mod.py").write_text("# cached")
            (Path(tmpdir) / "real.py").write_text("# real")
            planner = EditPlanner(working_dir=tmpdir)
            files = planner._get_project_files()
            assert "real.py" in files
            assert not any("__pycache__" in f for f in files)
