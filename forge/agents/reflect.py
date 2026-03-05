"""Self-reflecting agent: reviews its own output before presenting to the user.

After generating a response, the agent reviews it for:
- Factual errors or hallucinations
- Incomplete answers
- Code bugs
- Missing context

If issues are found, the agent revises its response. This pattern
significantly improves output quality, especially for smaller local
models that are more prone to errors.

The overhead is one additional LLM call per response, but the quality
improvement is substantial — similar to the difference between a
first draft and a reviewed answer.
"""

from __future__ import annotations

import re
from typing import Any

from forge.agents.base import BaseAgent, AgentConfig
from forge.agents.permissions import PermissionManager
from forge.llm.client import OllamaClient
from forge.utils.logging import get_logger

log = get_logger("agents.reflect")

# Limits
MAX_REVISIONS = 5  # hard cap on revision rounds
MIN_REVISIONS = 0  # 0 = reflection disabled
MIN_RESPONSE_LENGTH = 30  # responses shorter than this skip review
MAX_QUESTION_LENGTH = 500  # truncate question context in review prompts
MAX_RESPONSE_LENGTH = 2000  # truncate response context in review prompts
MAX_REVIEW_FEEDBACK_LENGTH = 500  # truncate review feedback in revision prompts

# LGTM detection — word boundary match (not embedded in other words)
LGTM_PATTERN = re.compile(r"\bLGTM\b", re.IGNORECASE)

REVIEW_PROMPT_SHORT = """Review this short response for accuracy and completeness.

User's question: {question}

Agent's response:
{response}

If the response is good, reply with exactly: LGTM
If there are issues, describe them briefly.
"""

REVIEW_PROMPT_CODE = """Review this response containing code. Check for:
1. Syntax errors, typos, or logic bugs
2. Missing imports or undefined variables
3. Edge cases not handled
4. Whether the code actually solves the stated problem

User's question: {question}

Agent's response:
{response}

If the response is good, reply with exactly: LGTM
If there are issues, describe them briefly and provide a corrected version.
"""

REVIEW_PROMPT_GENERAL = """Review this response for quality. Check for:
1. Factual accuracy — are statements correct?
2. Completeness — does it fully answer the question?
3. Code correctness — are there bugs, syntax errors, or logic issues?
4. Clarity — is it clear and well-organized?

User's question: {question}

Agent's response:
{response}

If the response is good, reply with exactly: LGTM
If there are issues, describe them briefly and provide a corrected version.
"""


class ReflectiveAgent(BaseAgent):
    """Agent that self-reviews responses before presenting them.

    Uses a generate-then-critique pattern:
    1. Generate initial response (normal chat)
    2. Review the response for issues
    3. If issues found, generate a revised response
    4. Present the final response to the user

    Usage:
        agent = ReflectiveAgent(client)
        response = agent.chat("Write a function to merge two sorted lists")
        # The response has been self-reviewed and corrected if needed
    """

    def __init__(
        self,
        client: OllamaClient,
        config: AgentConfig | None = None,
        working_dir: str = ".",
        permissions: PermissionManager | None = None,
        max_revisions: int = 1,
    ):
        super().__init__(
            client=client,
            config=config,
            working_dir=working_dir,
            permissions=permissions,
        )
        self.max_revisions = min(max(max_revisions, MIN_REVISIONS), MAX_REVISIONS)
        if max_revisions != self.max_revisions:
            log.warning("max_revisions clamped from %d to %d", max_revisions, self.max_revisions)
        self._review_count = 0
        self._revision_count = 0
        self._issue_categories: dict[str, int] = {}  # tracks issue types found

    def chat(self, user_message: str) -> str:
        """Chat with self-reflection on the response."""
        # Generate initial response
        response = super().chat(user_message)

        # Don't review very short responses or error messages
        if len(response) < MIN_RESPONSE_LENGTH or response.startswith("LLM error:"):
            return response

        # Review and potentially revise (up to max_revisions times)
        for _ in range(self.max_revisions):
            revised = self._review_and_revise(user_message, response)
            if revised == response:
                break  # Reviewer approved, no more changes needed
            response = revised

        # Update the stored message with the final version
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] = response

        return response

    def _select_review_prompt(self, response: str) -> str:
        """Select the most appropriate review prompt based on response content."""
        # Check for code content
        if "```" in response or "def " in response or "class " in response:
            return REVIEW_PROMPT_CODE
        # Short responses get a simpler review
        if len(response) < 200:
            return REVIEW_PROMPT_SHORT
        return REVIEW_PROMPT_GENERAL

    @staticmethod
    def _categorize_issues(review_text: str) -> list[str]:
        """Extract issue categories from review text."""
        categories = []
        text_lower = review_text.lower()
        if re.search(r"(incorrect|wrong|inaccurate|factual|false)", text_lower):
            categories.append("factual")
        if re.search(r"(incomplete|missing|doesn.t address|partial)", text_lower):
            categories.append("completeness")
        if re.search(r"(bug|syntax|error|typo|undefined|import)", text_lower):
            categories.append("code_error")
        if re.search(r"(unclear|confusing|hard to follow|readab)", text_lower):
            categories.append("clarity")
        return categories or ["general"]

    def _review_and_revise(self, question: str, response: str) -> str:
        """Review a response and revise if needed.

        Selects an appropriate review prompt based on response content
        and categorizes any issues found for tracking.
        """
        self._review_count += 1

        prompt_template = self._select_review_prompt(response)
        review_prompt = prompt_template.format(
            question=question[:MAX_QUESTION_LENGTH],
            response=response[:MAX_RESPONSE_LENGTH],
        )

        review_result = self.client.generate(
            prompt=review_prompt,
            system=(
                "You are a quality reviewer. Be concise. "
                "Say LGTM if the response is good. "
                "Otherwise, provide a corrected version."
            ),
            temperature=0.2,
        )

        review_text = review_result.get("response", "").strip()

        # Check if review approves the response
        if not review_text or LGTM_PATTERN.search(review_text):
            log.debug("Self-review: LGTM (no changes needed)")
            return response

        # Categorize the issues found
        categories = self._categorize_issues(review_text)
        for cat in categories:
            self._issue_categories[cat] = self._issue_categories.get(cat, 0) + 1

        # Review found issues — the review_text contains the correction
        log.info("Self-review found issues (%s), revising response", ", ".join(categories))
        self._revision_count += 1

        # Generate a revised response with targeted feedback
        revised_result = self.client.generate(
            prompt=(
                f"Original question: {question[:MAX_QUESTION_LENGTH]}\n\n"
                f"Your initial answer had these issues ({', '.join(categories)}):\n"
                f"{review_text[:MAX_REVIEW_FEEDBACK_LENGTH]}\n\n"
                f"Please provide a corrected, complete answer addressing all issues."
            ),
            system=self._system_prompt,
            temperature=self.config.temperature,
        )

        return revised_result.get("response", response)

    def get_stats(self) -> dict[str, Any]:
        """Get stats including reflection metrics and issue categories."""
        stats = super().get_stats()
        stats["reflection"] = {
            "reviews": self._review_count,
            "revisions": self._revision_count,
            "revision_rate": (
                round(self._revision_count / max(1, self._review_count), 2)
            ),
            "issue_categories": dict(self._issue_categories),
        }
        return stats
