"""Cascading agent: auto-switches to a bigger model when stuck.

Inspired by the "architect mode" pattern — uses a smaller, faster model for
routine work and escalates to a larger, more capable model when:
- The agent produces low-quality responses (short, repetitive, or error-laden)
- Tool calls fail repeatedly
- The user explicitly requests deeper reasoning

The cascade chain is determined by the hardware profile:
- compact: 3b → 7b
- standard: 7b → 14b
- workstation: 14b → 32b
- high_memory: 32b → 70b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.agents.base import BaseAgent, AgentConfig
from forge.llm.client import OllamaClient
from forge.llm.models import MODEL_CATALOGUE, get_models_that_fit
from forge.utils.logging import get_logger

log = get_logger("agents.cascade")

# Number of consecutive poor responses before escalating
ESCALATION_THRESHOLD = 3
MIN_ESCALATION_THRESHOLD = 1
MAX_ESCALATION_THRESHOLD = 20

# Minimum response length to be considered "useful"
MIN_USEFUL_LENGTH = 50

# Limits
MAX_CONSECUTIVE_POOR = 50  # hard cap to avoid unbounded counter

# Patterns that indicate the model is stuck
STUCK_PATTERNS = [
    "I don't know",
    "I'm not sure",
    "I cannot",
    "I can't",
    "I apologize, but",
    "As an AI language model",
    "I don't have the ability",
]


@dataclass
class CascadeConfig:
    """Configuration for model cascading."""

    primary_model: str = ""      # default model (auto-detected)
    escalation_model: str = ""   # bigger model to switch to when stuck
    escalation_threshold: int = ESCALATION_THRESHOLD  # validated in CascadeAgent.__init__
    auto_deescalate: bool = True  # return to primary after successful escalation


class CascadeAgent(BaseAgent):
    """Agent that automatically escalates to a larger model when stuck.

    Usage:
        agent = CascadeAgent(client, cascade_config=CascadeConfig(
            primary_model="qwen2.5-coder:7b",
            escalation_model="qwen2.5-coder:14b",
        ))

    The agent tracks response quality and automatically switches to the
    escalation model when it detects the primary model is struggling.
    After a successful response from the escalation model, it optionally
    returns to the primary model for efficiency.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: AgentConfig | None = None,
        cascade_config: CascadeConfig | None = None,
        working_dir: str = ".",
    ):
        super().__init__(client=client, config=config, working_dir=working_dir)
        self.cascade = cascade_config or CascadeConfig()
        # Validate escalation threshold
        self.cascade.escalation_threshold = min(
            max(self.cascade.escalation_threshold, MIN_ESCALATION_THRESHOLD),
            MAX_ESCALATION_THRESHOLD,
        )
        self._consecutive_poor = 0
        self._is_escalated = False
        self._primary_model = self.cascade.primary_model or client.model
        self._escalation_model = self.cascade.escalation_model
        # Escalation metrics
        self._escalation_count = 0
        self._deescalation_count = 0
        self._escalation_successes = 0  # escalation led to good response

        # Warn if primary and escalation are the same
        if self._primary_model and self._escalation_model and self._primary_model == self._escalation_model:
            log.warning("Primary and escalation models are the same: %s", self._primary_model)

    def chat(self, user_message: str) -> str:
        """Chat with automatic model escalation on poor responses."""
        response = super().chat(user_message)

        if self._is_poor_response(response):
            self._consecutive_poor = min(self._consecutive_poor + 1, MAX_CONSECUTIVE_POOR)
            log.info(
                "Poor response detected (%d/%d before escalation)",
                self._consecutive_poor,
                self.cascade.escalation_threshold,
            )

            if self._consecutive_poor >= self.cascade.escalation_threshold:
                return self._escalate_and_retry(user_message)
        else:
            self._consecutive_poor = 0

            # De-escalate after successful response from bigger model
            if self._is_escalated:
                self._escalation_successes += 1
                if self.cascade.auto_deescalate:
                    self._deescalate()

        return response

    def _is_poor_response(self, response: str) -> bool:
        """Detect if a response indicates the model is struggling."""
        if not response:
            return True

        if len(response.strip()) < MIN_USEFUL_LENGTH:
            return True

        response_lower = response.lower()
        for pattern in STUCK_PATTERNS:
            if pattern.lower() in response_lower:
                return True

        return False

    def _escalate_and_retry(self, user_message: str) -> str:
        """Switch to the larger model and retry the last message.

        Checks model availability before escalating — if the escalation
        model isn't available and can't be pulled, stays on the primary model.
        """
        if not self._escalation_model:
            log.warning("No escalation model configured")
            return self.messages[-1].get("content", "") if self.messages else ""

        # Check if escalation model is available before switching
        if not self._is_model_available(self._escalation_model):
            log.warning(
                "Escalation model %s not available — staying on %s",
                self._escalation_model, self._primary_model,
            )
            self._consecutive_poor = 0  # Reset to avoid infinite escalation attempts
            return self.messages[-1].get("content", "") if self.messages else ""

        log.info("Escalating from %s to %s", self._primary_model, self._escalation_model)

        # Switch model
        self.client.switch_model(self._escalation_model)
        self._is_escalated = True
        self._consecutive_poor = 0
        self._escalation_count += 1

        # Remove the poor response and retry
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()

        return super().chat(user_message)

    def _is_model_available(self, model: str) -> bool:
        """Check if a model is locally available (without pulling)."""
        try:
            available = self.client.list_models()
            available_names = {m.get("name", "") for m in available}
            # Also check without tag (e.g., "qwen2.5-coder:7b" matches "qwen2.5-coder:7b")
            available_bases = {n.split(":")[0] for n in available_names}
            model_base = model.split(":")[0]
            return model in available_names or model_base in available_bases
        except Exception:
            return False

    def _deescalate(self) -> None:
        """Return to the primary (smaller) model."""
        log.info("De-escalating from %s to %s", self._escalation_model, self._primary_model)
        self.client.switch_model(self._primary_model)
        self._is_escalated = False
        self._deescalation_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get stats including cascade information."""
        stats = super().get_stats()
        stats["cascade"] = {
            "primary_model": self._primary_model,
            "escalation_model": self._escalation_model,
            "is_escalated": self._is_escalated,
            "consecutive_poor": self._consecutive_poor,
            "escalation_count": self._escalation_count,
            "deescalation_count": self._deescalation_count,
            "escalation_successes": self._escalation_successes,
            "escalation_success_rate": round(
                self._escalation_successes / max(1, self._escalation_count), 2
            ),
        }
        return stats


def auto_cascade_config(gpu_gb: float) -> CascadeConfig:
    """Generate cascade config based on available GPU memory.

    Selects the best primary and escalation models that fit in GPU memory.
    """
    fitting = get_models_that_fit(gpu_gb)
    coding_models = [m for m in fitting if m.category == "coding"]

    if len(coding_models) >= 2:
        # Use the second-largest as primary, largest as escalation
        return CascadeConfig(
            primary_model=coding_models[1].name,
            escalation_model=coding_models[0].name,
        )
    elif len(coding_models) == 1:
        # Only one coding model fits — use general models for escalation
        general_models = [m for m in fitting if m.category == "general"]
        escalation = general_models[0].name if general_models else ""
        return CascadeConfig(
            primary_model=coding_models[0].name,
            escalation_model=escalation,
        )
    else:
        return CascadeConfig()
