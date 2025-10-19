"""Agent that translates exceptions into helpful troubleshooting guidance."""
from __future__ import annotations

from typing import Any

from .base import Agent, AgentError


class DebuggingAgent(Agent):
    """Provides friendly error messages and potential fixes."""

    name = "debugging"

    def run(self, *, error: Exception | None = None, **_: Any) -> str:
        if error is None:
            return "No errors to debug."

        if isinstance(error, AgentError):
            return self._handle_agent_error(error)

        return (
            "An unexpected error occurred. Please review the stack trace and ensure "
            "that the input file is well-formed."
        )

    def _handle_agent_error(self, error: AgentError) -> str:
        agent_info = f" in agent '{error.agent_name}'" if error.agent_name else ""
        return f"Encountered a recoverable error{agent_info}: {error}."
