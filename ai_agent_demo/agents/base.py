"""Common base classes and protocols for agent components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentContext(Dict[str, Any]):
    """Dictionary-like context shared across agents in the pipeline."""


class Agent(ABC):
    """Abstract base class for all agents in the demo pipeline."""

    name: str = "agent"

    def __init__(self) -> None:
        self.context: AgentContext = AgentContext()

    def update_context(self, **kwargs: Any) -> None:
        """Store key-value pairs that downstream agents can reuse."""
        self.context.update(kwargs)

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the agent logic."""


class AgentError(Exception):
    """Custom exception raised by agents when a recoverable error occurs."""

    def __init__(self, message: str, *, agent_name: str | None = None) -> None:
        super().__init__(message)
        self.agent_name = agent_name
