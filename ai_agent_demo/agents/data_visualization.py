"""Agent that creates matplotlib charts from tabular data."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd

from .base import Agent, AgentError


class DataVisualizationAgent(Agent):
    """Produces histogram and scatter plot visualizations for numeric data."""

    name = "data_visualization"

    def run(
        self,
        *,
        dataframe: pd.DataFrame | None = None,
        output_dir: str | None = None,
        **_: Any,
    ) -> List[Path]:
        if dataframe is None:
            raise AgentError("DataFrame required for visualization.", agent_name=self.name)

        if output_dir is None:
            raise AgentError("Output directory is required for visualization.", agent_name=self.name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: List[Path] = []
        numeric_columns = dataframe.select_dtypes(include="number")

        if numeric_columns.empty:
            raise AgentError("No numeric columns available for visualization.", agent_name=self.name)

        for column in numeric_columns.columns:
            fig, ax = plt.subplots()
            dataframe[column].plot.hist(ax=ax, title=f"Distribution of {column}")
            ax.set_xlabel(column)
            hist_path = output_path / f"hist_{column}.png"
            fig.savefig(hist_path, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(hist_path)

        if numeric_columns.shape[1] > 1:
            columns = numeric_columns.columns[:2]
            fig, ax = plt.subplots()
            ax.scatter(dataframe[columns[0]], dataframe[columns[1]])
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_title(f"Scatter plot: {columns[0]} vs {columns[1]}")
            scatter_path = output_path / f"scatter_{columns[0]}_{columns[1]}.png"
            fig.savefig(scatter_path, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(scatter_path)

        self.update_context(visualizations=[str(path) for path in saved_files])
        return saved_files
