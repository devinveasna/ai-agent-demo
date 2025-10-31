"""Agent that translates exceptions into helpful troubleshooting guidance."""
from __future__ import annotations

import json
from typing import Any

from .base import Agent, AgentError
from ..llm import LLMSettings


class DebuggingAgent(Agent):
    """Provides friendly error messages and potential fixes."""

    name = "debugging"

    def __init__(
        self,
        *,
        llm_settings: LLMSettings | None = None,
        llm_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.llm_settings = llm_settings
        self.llm_temperature = llm_temperature
        self._llm_client, self._init_warnings = (
            self.llm_settings.create_client()
            if self.llm_settings is not None
            else (None, [])
        )

    def run(self, *, error: Exception | None = None, **_: Any) -> str:
        if error is None:
            return "No errors to debug."

        if isinstance(error, AgentError):
            message = self._handle_agent_error(error)
        else:
            message = (
                "An unexpected error occurred. Please review the stack trace and ensure "
                "that the input file is well-formed."
            )

        if self._llm_client is None or self.llm_settings is None:
            if self._init_warnings:
                message += "\n\nLLM debugging unavailable: " + " ".join(self._init_warnings)
            return message

        payload = {
            "agent": getattr(error, "agent_name", None),
            "error_type": type(error).__name__,
            "message": str(error),
        }

        try:  # pragma: no cover - requires external LLM
            response = self._llm_client.chat.completions.create(
                model=self.llm_settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that helps developers debug a data analysis pipeline. "
                            "Offer concrete troubleshooting steps based on the error details."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Error context:" + json.dumps(payload, default=str),
                    },
                ],
                temperature=self.llm_temperature,
            )
            llm_message = response.choices[0].message  # type: ignore[index]
            llm_content = getattr(llm_message, "content", None)
        except Exception as exc:  # pragma: no cover - network failures
            message += f"\n\nLLM debugging unavailable: {exc}"
        else:
            if llm_content:
                message += "\n\nLLM suggestion:\n" + llm_content.strip()
            elif self._init_warnings:
                message += "\n\nLLM debugging unavailable: " + " ".join(self._init_warnings)

        return message

    def _handle_agent_error(self, error: AgentError) -> str:
        agent_info = f" in agent '{error.agent_name}'" if error.agent_name else ""
        return f"Encountered a recoverable error{agent_info}: {error}."
