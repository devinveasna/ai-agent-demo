"""Agent that creates matplotlib charts from tabular data."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from .base import Agent, AgentError
from .chart_planning import ChartPlan, ChartSpec


class DataVisualizationAgent(Agent):
    """Produces charts based on a plan, with heuristic fallbacks."""

    name = "data_visualization"

    def run(
        self,
        *,
        dataframe: pd.DataFrame | None = None,
        output_dir: str | None = None,
        plan: ChartPlan | None = None,
        **_: Any,
    ) -> List[Path]:
        if dataframe is None:
            raise AgentError("DataFrame required for visualization.", agent_name=self.name)

        if output_dir is None:
            raise AgentError("Output directory is required for visualization.", agent_name=self.name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: List[Path] = []
        specs: Iterable[ChartSpec]
        if plan is not None and plan.charts:
            specs = plan.charts
        else:
            specs = self._default_plan(dataframe)

        for spec in specs:
            try:
                artifact = self._render_chart(dataframe=dataframe, spec=spec, output_dir=output_path)
            except AgentError:
                raise
            except Exception as exc:  # pragma: no cover - plotting edge cases
                raise AgentError(str(exc), agent_name=self.name) from exc
            if artifact is not None:
                saved_files.append(artifact)

        if not saved_files:
            raise AgentError("No charts could be generated for the provided plan.", agent_name=self.name)

        plan_source = plan.source if plan is not None else "default"
        self.update_context(visualizations=[str(path) for path in saved_files], plan_source=plan_source)
        return saved_files

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_plan(self, dataframe: pd.DataFrame) -> Iterable[ChartSpec]:
        numeric_columns = dataframe.select_dtypes(include="number")
        if numeric_columns.empty:
            raise AgentError("No numeric columns available for visualization.", agent_name=self.name)

        charts: List[ChartSpec] = []
        for column in numeric_columns.columns:
            charts.append(
                ChartSpec(
                    chart_type="histogram",
                    x=column,
                    title=f"Distribution of {column}",
                    options={},
                )
            )

        if numeric_columns.shape[1] > 1:
            columns = numeric_columns.columns[:2]
            charts.append(
                ChartSpec(
                    chart_type="scatter",
                    x=columns[0],
                    y=columns[1],
                    title=f"Scatter plot: {columns[0]} vs {columns[1]}",
                    options={},
                )
            )

        return charts

    def _render_chart(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        if spec.x not in dataframe.columns:
            return None
        if spec.y is not None and spec.y not in dataframe.columns:
            return None

        chart_type = spec.chart_type.lower()
        if chart_type == "histogram":
            return self._render_histogram(dataframe=dataframe, spec=spec, output_dir=output_dir)
        if chart_type == "scatter":
            return self._render_scatter(dataframe=dataframe, spec=spec, output_dir=output_dir)
        if chart_type == "line":
            return self._render_line(dataframe=dataframe, spec=spec, output_dir=output_dir)
        if chart_type == "bar":
            return self._render_bar(dataframe=dataframe, spec=spec, output_dir=output_dir)
        if chart_type == "box":
            return self._render_box(dataframe=dataframe, spec=spec, output_dir=output_dir)
        return None

    def _render_histogram(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        series = dataframe[spec.x].dropna()
        if series.empty:
            return None
        bins = spec.options.get("bins") if isinstance(spec.options, dict) else None
        color = spec.options.get("color") if isinstance(spec.options, dict) else None
        fig, ax = plt.subplots()
        series.plot.hist(ax=ax, bins=bins, color=color)
        ax.set_title(spec.title or f"Distribution of {spec.x}")
        ax.set_xlabel(spec.x)
        ax.set_ylabel("Frequency")
        path = output_dir / f"hist_{spec.x}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _render_scatter(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        if spec.y is None:
            return None
        subset = dataframe[[spec.x, spec.y]].dropna()
        if subset.empty:
            return None
        color = None
        alpha = 0.8
        if isinstance(spec.options, dict):
            color = spec.options.get("color")
            try:
                alpha = float(spec.options.get("alpha", alpha))
            except (TypeError, ValueError):
                alpha = 0.8
        fig, ax = plt.subplots()
        ax.scatter(subset[spec.x], subset[spec.y], c=color, alpha=alpha)
        ax.set_xlabel(spec.x)
        ax.set_ylabel(spec.y)
        ax.set_title(spec.title or f"Scatter plot: {spec.x} vs {spec.y}")
        path = output_dir / f"scatter_{spec.x}_{spec.y}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _render_line(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots()
        color = None
        linewidth = None
        linestyle = None
        if isinstance(spec.options, dict):
            color = spec.options.get("color")
            linewidth = spec.options.get("linewidth")
            linestyle = spec.options.get("linestyle")
        if spec.y is not None:
            subset = dataframe[[spec.x, spec.y]].dropna()
            if subset.empty:
                plt.close(fig)
                return None
            ax.plot(subset[spec.x], subset[spec.y], color=color, linewidth=linewidth, linestyle=linestyle)
            ax.set_ylabel(spec.y)
        else:
            series = dataframe[spec.x].dropna()
            if series.empty:
                plt.close(fig)
                return None
            ax.plot(series.index, series.values, color=color, linewidth=linewidth, linestyle=linestyle)
            ax.set_ylabel(spec.x)
        ax.set_xlabel(spec.x if spec.y is not None else "Index")
        ax.set_title(spec.title or (f"Line plot of {spec.x}" if spec.y is None else f"Line plot: {spec.x} vs {spec.y}"))
        path = output_dir / f"line_{spec.x}{'_' + spec.y if spec.y else ''}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _render_bar(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots()
        color = None
        top_n = None
        if isinstance(spec.options, dict):
            color = spec.options.get("color")
            top_n = spec.options.get("top_n")
        if spec.y is not None:
            subset = dataframe[[spec.x, spec.y]].dropna()
            if subset.empty:
                plt.close(fig)
                return None
            ax.bar(subset[spec.x], subset[spec.y], color=color)
            ax.set_ylabel(spec.y)
        else:
            counts = dataframe[spec.x].dropna().value_counts()
            if top_n:
                try:
                    counts = counts.head(int(top_n))
                except (TypeError, ValueError):
                    pass
            if counts.empty:
                plt.close(fig)
                return None
            counts.plot.bar(ax=ax, color=color)
            ax.set_ylabel("Count")
        ax.set_xlabel(spec.x)
        ax.set_title(spec.title or f"Bar chart of {spec.x}")
        path = output_dir / f"bar_{spec.x}{'_' + spec.y if spec.y else ''}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _render_box(
        self,
        *,
        dataframe: pd.DataFrame,
        spec: ChartSpec,
        output_dir: Path,
    ) -> Path | None:
        fig, ax = plt.subplots()
        if spec.y is not None:
            subset = dataframe[[spec.x, spec.y]].dropna()
            if subset.empty:
                plt.close(fig)
                return None
            subset.boxplot(column=spec.y, by=spec.x, ax=ax)
            ax.set_title(spec.title or f"Box plot of {spec.y} by {spec.x}")
            ax.set_xlabel(spec.x)
            ax.set_ylabel(spec.y)
            fig.suptitle("")
        else:
            series = dataframe[spec.x].dropna()
            if series.empty:
                plt.close(fig)
                return None
            ax.boxplot(series)
            ax.set_title(spec.title or f"Box plot of {spec.x}")
            ax.set_xlabel("Values")
            ax.set_ylabel(spec.x)
        path = output_dir / f"box_{spec.x}{'_' + spec.y if spec.y else ''}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
