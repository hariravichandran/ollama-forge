"""Tests for system prompt templates."""

import pytest

from forge.agents.prompts import (
    PROMPT_TEMPLATES,
    PromptTemplate,
    get_prompt,
    list_templates,
)


class TestPromptTemplates:
    """Tests for the prompt template system."""

    def test_templates_exist(self):
        """Should have at least 5 templates."""
        assert len(PROMPT_TEMPLATES) >= 5

    def test_all_templates_are_valid(self):
        """All templates should have required fields."""
        for name, tmpl in PROMPT_TEMPLATES.items():
            assert isinstance(tmpl, PromptTemplate)
            assert tmpl.name == name
            assert len(tmpl.description) > 0
            assert len(tmpl.template) > 0
            assert tmpl.category in ("general", "development", "research",
                                      "writing", "data", "operations", "education")

    def test_core_templates_present(self):
        """Core templates should be defined."""
        assert "coder" in PROMPT_TEMPLATES
        assert "reviewer" in PROMPT_TEMPLATES
        assert "researcher" in PROMPT_TEMPLATES
        assert "debugger" in PROMPT_TEMPLATES
        assert "writer" in PROMPT_TEMPLATES
        assert "devops" in PROMPT_TEMPLATES
        assert "tutor" in PROMPT_TEMPLATES

    def test_get_prompt_basic(self):
        """get_prompt should return a string."""
        prompt = get_prompt("coder")
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "developer" in prompt.lower() or "code" in prompt.lower()

    def test_get_prompt_with_variables(self):
        """get_prompt should fill in template variables."""
        prompt = get_prompt("coder", language="Python")
        assert "Python" in prompt

    def test_get_prompt_without_variables(self):
        """get_prompt with no variables should strip placeholders."""
        prompt = get_prompt("coder")
        assert "{language_note}" not in prompt
        assert "{project_note}" not in prompt

    def test_get_prompt_unknown_template(self):
        """Unknown template should return a default prompt."""
        prompt = get_prompt("nonexistent_template_xyz")
        assert isinstance(prompt, str)
        assert "helpful" in prompt.lower()

    def test_reviewer_no_variables(self):
        """Reviewer template has no variables."""
        tmpl = PROMPT_TEMPLATES["reviewer"]
        assert tmpl.variables == []
        prompt = get_prompt("reviewer")
        assert "review" in prompt.lower()

    def test_debugger_template(self):
        """Debugger should emphasize systematic approach."""
        prompt = get_prompt("debugger")
        assert "root cause" in prompt.lower()

    def test_list_templates(self):
        """list_templates should return formatted list."""
        templates = list_templates()
        assert len(templates) >= 5
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "category" in t
            assert "variables" in t
