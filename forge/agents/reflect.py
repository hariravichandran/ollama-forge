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

from typing import Any

from forge.agents.base import BaseAgent, AgentConfig
from forge.agents.permissions import PermissionManager
from forge.llm.client import OllamaClient
from forge.utils.logging import get_logger

log = get_logger("agents.reflect")

REVIEW_PROMPT = """Review this response for quality. Check for:
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
        self.max_revisions = max_revisions
        self._review_count = 0
        self._revision_count = 0

    def chat(self, user_message: str) -> str:
        """Chat with self-reflection on the response."""
        # Generate initial response
        initial_response = super().chat(user_message)

        # Don't review very short responses or error messages
        if len(initial_response) < 30 or initial_response.startswith("LLM error:"):
            return initial_response

        # Review and potentially revise
        final_response = self._review_and_revise(user_message, initial_response)
        if final_response != initial_response:
            # Update the stored message with the revised version
            if self.messages and self.messages[-1]["role"] == "assistant":
                self.messages[-1]["content"] = final_response

        return final_response

    def _review_and_revise(self, question: str, response: str) -> str:
        """Review a response and revise if needed."""
        self._review_count += 1

        review_prompt = REVIEW_PROMPT.format(
            question=question[:500],
            response=response[:2000],
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
        if "LGTM" in review_text.upper() or not review_text:
            log.debug("Self-review: LGTM (no changes needed)")
            return response

        # Review found issues — the review_text contains the correction
        log.info("Self-review found issues, revising response")
        self._revision_count += 1

        # Generate a revised response incorporating the review feedback
        revised_result = self.client.generate(
            prompt=(
                f"Original question: {question[:500]}\n\n"
                f"Your initial answer had these issues:\n{review_text[:500]}\n\n"
                f"Please provide a corrected, complete answer."
            ),
            system=self._system_prompt,
            temperature=self.config.temperature,
        )

        return revised_result.get("response", response)

    def get_stats(self) -> dict[str, Any]:
        """Get stats including reflection metrics."""
        stats = super().get_stats()
        stats["reflection"] = {
            "reviews": self._review_count,
            "revisions": self._revision_count,
            "revision_rate": (
                round(self._revision_count / max(1, self._review_count), 2)
            ),
        }
        return stats
