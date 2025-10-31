"""Utilities for configuring LLM access across agents."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class LLMSettings:
    """Shared connection details for agents that optionally call an LLM."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float | None = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_client(self) -> Tuple[Any | None, List[str]]:
        """Instantiate an OpenAI-compatible client if possible.

        Returns a tuple of ``(client, warnings)`` so that callers can surface
        configuration issues (e.g., missing dependencies) without raising.
        """

        warnings: List[str] = []

        if not self.model:
            warnings.append("LLM model not specified; skipping client creation.")
            return None, warnings

        if OpenAI is None:
            warnings.append(
                "openai package is unavailable; install `openai` to enable LLM-backed agents."
            )
            return None, warnings

        api_key = self.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        kwargs: Dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_kwargs:
            kwargs.update(self.extra_kwargs)

        try:  # pragma: no cover - depends on external client availability
            client = OpenAI(api_key=api_key, **kwargs)
        except Exception as exc:  # pragma: no cover - client construction failure
            warnings.append(f"Failed to initialise LLM client: {exc}")
            return None, warnings

        return client, warnings


def describe_llm_endpoint(settings: LLMSettings | None) -> str:
    """Return a human-readable description of the configured endpoint."""

    if settings is None:
        return "LLM disabled"

    target = settings.base_url or "https://api.openai.com"
    return f"model={settings.model} @ {target}"

