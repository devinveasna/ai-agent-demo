"""Agent that proposes visualization plans, optionally via an LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, Iterable, List

import pandas as pd

from .base import Agent, AgentError
from .data_analysis import AnalysisReport

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


@dataclass
class ChartSpec:
    """Structured representation of a single chart to generate."""

    chart_type: str
    x: str
    y: str | None = None
    title: str | None = None
    description: str | None = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartPlan:
    """Collection of chart specifications and metadata about their origin."""

    charts: List[ChartSpec]
    source: str
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.charts:
            return "No charts were planned."

        lines = [f"### Chart plan (source: {self.source})", ""]
        for spec in self.charts:
            title = spec.title or f"{spec.chart_type.title()} of {spec.x}" + (f" vs {spec.y}" if spec.y else "")
            lines.append(f"- **{title}**")
            detail_parts = [f"type=`{spec.chart_type}`", f"x=`{spec.x}`"]
            if spec.y:
                detail_parts.append(f"y=`{spec.y}`")
            if spec.description:
                detail_parts.append(spec.description)
            if spec.options:
                detail_parts.append(f"options={spec.options}")
            lines.append(f"  - {'; '.join(detail_parts)}")
        if self.warnings:
            lines.extend(["", "#### Planner warnings", ""])
            lines.extend([f"- {warning}" for warning in self.warnings])
        return "\n".join(lines)


class ChartPlanningAgent(Agent):
    """Generates chart plans using heuristics or an LLM prompt."""

    name = "chart_planning"

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        temperature: float = 0.2,
        max_charts: int = 6,
    ) -> None:
        super().__init__()
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_charts = max_charts
        self._llm_client: Any | None = None
        if llm_model and OpenAI is not None:
            try:  # pragma: no cover - network interactions are optional
                self._llm_client = OpenAI()
            except Exception:  # pragma: no cover
                self._llm_client = None

    def run(
        self,
        *,
        dataframe: pd.DataFrame | None = None,
        analysis_report: AnalysisReport | None = None,
        **_: Any,
    ) -> ChartPlan:
        if dataframe is None:
            raise AgentError("DataFrame required for chart planning.", agent_name=self.name)

        if self.llm_model:
            plan = self._attempt_llm_plan(dataframe=dataframe, analysis_report=analysis_report)
        else:
            plan = None

        if plan is None or (plan.source == "heuristic" and not plan.charts):
            fallback = self._heuristic_plan(dataframe=dataframe)
            if plan is not None and plan.warnings:
                fallback.warnings.extend(plan.warnings)
            plan = fallback

        self.update_context(plan=plan)
        return plan

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _attempt_llm_plan(
        self,
        *,
        dataframe: pd.DataFrame,
        analysis_report: AnalysisReport | None,
    ) -> ChartPlan | None:
        if self._llm_client is None:
            warning = "LLM model configured but OpenAI client unavailable; falling back to heuristics."
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        schema_lines = [
            f"- {column}: {str(dtype)}"
            for column, dtype in dataframe.dtypes.items()
        ]
        sample_rows = self._make_json_safe(dataframe.head().to_dict(orient="records"))
        analysis_summary: Dict[str, Any] = (
            self._make_json_safe(analysis_report.summary)
            if analysis_report is not None
            else {}
        )

        instructions = dedent(
            f"""
            You are a senior data visualization expert. Propose a concise plan of charts to help a scientist explore the dataset.
            Return strict JSON with the shape {{"charts": [...]}} and nothing else.
            Use at most {self.max_charts} charts. Choose chart types from [histogram, scatter, line, bar, box].
            Prefer numeric-only charts for numeric columns and categorical summaries for categorical columns.
            Each chart must include fields: chart_type, x, optional y, optional title, optional description, optional options dict.
            If you cannot find a valid chart, return an empty list. Avoid referencing columns that do not exist.
            """
        ).strip()

        user_payload = {
            "schema": schema_lines,
            "analysis_summary": analysis_summary,
            "sample_rows": sample_rows,
        }

        try:  # pragma: no cover - requires network access
            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": "Plan helpful charts for this dataset:" + json.dumps(user_payload),
                    },
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover
            warning = f"LLM planning failed: {exc}"
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        try:
            message = response.choices[0].message  # type: ignore[index]
            content = getattr(message, "content", None)
        except Exception as exc:  # pragma: no cover
            warning = f"Unexpected LLM response structure: {exc}"
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        if not content:
            warning = "LLM returned empty content; using heuristic plan."
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover
            warning = f"Failed to parse LLM JSON: {exc}"
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        charts = list(self._parse_chart_specs(payload.get("charts", []), dataframe))
        if not charts:
            warning = "LLM did not yield valid charts; using heuristic plan."
            return ChartPlan(charts=[], source="heuristic", warnings=[warning])

        return ChartPlan(charts=charts, source="llm")

    def _parse_chart_specs(
        self,
        chart_payload: Iterable[Dict[str, Any]],
        dataframe: pd.DataFrame,
    ) -> Iterable[ChartSpec]:
        available_columns = set(dataframe.columns)
        for raw in chart_payload:
            chart_type = str(raw.get("chart_type", "")).strip().lower()
            x = raw.get("x")
            if not chart_type or x not in available_columns:
                continue
            y = raw.get("y")
            if y is not None and y not in available_columns:
                continue
            description = raw.get("description")
            options_raw = raw.get("options")
            options = options_raw if isinstance(options_raw, dict) else {}
            title = raw.get("title")
            yield ChartSpec(
                chart_type=chart_type,
                x=x,
                y=y,
                title=title,
                description=description,
                options=dict(options),
            )

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------
    def _heuristic_plan(self, *, dataframe: pd.DataFrame) -> ChartPlan:
        numeric = dataframe.select_dtypes(include="number")
        categorical = dataframe.select_dtypes(exclude="number")

        charts: List[ChartSpec] = []
        for column in numeric.columns:
            charts.append(
                ChartSpec(
                    chart_type="histogram",
                    x=column,
                    title=f"Distribution of {column}",
                    description="Auto-generated histogram for numeric column.",
                    options={"bins": min(40, max(10, int(len(dataframe) ** 0.5)))},
                )
            )

        if numeric.shape[1] > 1:
            first, second = numeric.columns[:2]
            charts.append(
                ChartSpec(
                    chart_type="scatter",
                    x=first,
                    y=second,
                    title=f"Scatter plot: {first} vs {second}",
                    description="Compare two numeric features to inspect correlation.",
                    options={"alpha": 0.7},
                )
            )

        if not categorical.empty:
            top_cat = categorical.columns[0]
            charts.append(
                ChartSpec(
                    chart_type="bar",
                    x=top_cat,
                    title=f"Category distribution: {top_cat}",
                    description="Bar chart showing counts per category.",
                    options={"top_n": 20},
                )
            )

        return ChartPlan(charts=charts, source="heuristic")

    def _make_json_safe(self, value: Any) -> Any:
        """Convert nested structures into JSON-serialisable primitives."""

        try:
            json.dumps(value)
            return value
        except TypeError:
            pass

        if isinstance(value, dict):
            return {str(key): self._make_json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                scalar = value.item()  # type: ignore[call-arg]
                return self._make_json_safe(scalar)
            except Exception:
                return str(value)
        return str(value)
