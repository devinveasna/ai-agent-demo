"""Agent that creates matplotlib charts from tabular data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .base import Agent, AgentError


@dataclass(frozen=True)
class VisualizationRequest:
    """Represents a chart that the user would like to generate."""

    chart_type: str
    columns: Sequence[str]


class DataVisualizationAgent(Agent):
    """Produces histogram and scatter plot visualizations for numeric data."""

    name = "data_visualization"

    SUPPORTED_CHARTS = {"histogram", "scatter"}

    def run(
        self,
        *,
        dataframe: pd.DataFrame | None = None,
        output_dir: str | None = None,
        requests: Sequence[VisualizationRequest] | None = None,
        **_: Any,
    ) -> List[Path]:
        if dataframe is None:
            raise AgentError("DataFrame required for visualization.", agent_name=self.name)

        if output_dir is None:
            raise AgentError("Output directory is required for visualization.", agent_name=self.name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        numeric_columns = dataframe.select_dtypes(include="number")

        if numeric_columns.empty:
            raise AgentError("No numeric columns available for visualization.", agent_name=self.name)

        if requests is not None:
            charts_to_render = list(
                self._validate_requests(
                    requests=requests, numeric_columns=numeric_columns.columns
                )
            )
        else:
            charts_to_render = list(self._default_requests(numeric_columns.columns))

        saved_files = [
            self._render_chart(request=request, dataframe=dataframe, output_path=output_path)
            for request in charts_to_render
        ]

        self.update_context(visualizations=[str(path) for path in saved_files])
        return saved_files

    def _render_chart(
        self,
        *,
        request: VisualizationRequest,
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        if request.chart_type == "histogram":
            column = request.columns[0]
            fig, ax = plt.subplots()
            dataframe[column].plot.hist(ax=ax, title=f"Distribution of {column}")
            ax.set_xlabel(column)
            file_path = output_path / f"hist_{column}.png"
        elif request.chart_type == "scatter":
            x_col, y_col = request.columns[:2]
            fig, ax = plt.subplots()
            ax.scatter(dataframe[x_col], dataframe[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter plot: {x_col} vs {y_col}")
            file_path = output_path / f"scatter_{x_col}_{y_col}.png"
        else:  # pragma: no cover - safeguarded by validation
            raise AgentError(
                f"Unsupported chart type '{request.chart_type}'.",
                agent_name=self.name,
            )

        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        return file_path

    def _validate_requests(
        self,
        *,
        requests: Sequence[VisualizationRequest],
        numeric_columns: Sequence[str],
    ) -> Iterable[VisualizationRequest]:
        numeric_set = set(numeric_columns)
        validated: list[VisualizationRequest] = []

        for request in requests:
            if request.chart_type not in self.SUPPORTED_CHARTS:
                raise AgentError(
                    f"Unsupported chart type '{request.chart_type}'.",
                    agent_name=self.name,
                )

            if request.chart_type == "histogram":
                if len(request.columns) != 1:
                    raise AgentError(
                        "Histogram requests must contain exactly one column.",
                        agent_name=self.name,
                    )
            elif request.chart_type == "scatter":
                if len(request.columns) != 2:
                    raise AgentError(
                        "Scatter plot requests must contain exactly two columns.",
                        agent_name=self.name,
                    )

            missing_columns = [col for col in request.columns if col not in numeric_set]
            if missing_columns:
                missing = ", ".join(missing_columns)
                raise AgentError(
                    f"Columns not suitable for visualization: {missing}.",
                    agent_name=self.name,
                )

            validated.append(request)

        return validated

    def _default_requests(self, numeric_columns: Sequence[str]) -> Iterable[VisualizationRequest]:
        for column in numeric_columns:
            yield VisualizationRequest(chart_type="histogram", columns=[column])

        if len(numeric_columns) > 1:
            first, second = numeric_columns[:2]
            yield VisualizationRequest(chart_type="scatter", columns=[first, second])
