"""Agent responsible for extracting structured data from local files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import Agent, AgentError


class DataExtractionAgent(Agent):
    """Loads tabular data from CSV or simple text files into a DataFrame."""

    name = "data_extraction"

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt"}

    def run(self, *, file_path: str, **_: Any) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise AgentError(f"File not found: {file_path}", agent_name=self.name)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise AgentError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported extensions: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}",
                agent_name=self.name,
            )

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else None
            frame = pd.read_csv(path, delimiter=delimiter)

        if frame.empty:
            raise AgentError("The input dataset is empty.", agent_name=self.name)

        self.update_context(dataframe=frame, source_path=str(path))
        return frame
