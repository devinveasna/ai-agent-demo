"""High-level orchestration for the AI agent demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import pandas as pd

from .agents.base import Agent
from .agents.chart_planning import ChartPlan, ChartPlanningAgent
from .agents.data_analysis import AnalysisReport, DataAnalysisAgent
from .agents.data_extraction import DataExtractionAgent
from .agents.data_visualization import DataVisualizationAgent
from .agents.debugging import DebuggingAgent


@dataclass
class OrchestratorResult:
    dataframe_preview: str
    analysis_report: AnalysisReport | None
    chart_plan: ChartPlan | None
    visualization_paths: List[str]
    debug_message: str | None = None


@dataclass
class AgentOrchestrator:
    """Runs the multi-agent data-to-visualization workflow."""

    agents: Sequence[Agent] = field(
        default_factory=lambda: (
            DataExtractionAgent(),
            DataAnalysisAgent(),
            ChartPlanningAgent(),
            DataVisualizationAgent(),
        )
    )
    debugger: DebuggingAgent = field(default_factory=DebuggingAgent)

    def run(self, *, file_path: str, output_dir: str) -> OrchestratorResult:
        dataframe = None
        analysis_report: AnalysisReport | None = None
        chart_plan: ChartPlan | None = None
        visualization_paths: List[str] = []

        try:
            for agent in self.agents:
                if isinstance(agent, DataExtractionAgent):
                    dataframe = agent.run(file_path=file_path)
                elif isinstance(agent, DataAnalysisAgent):
                    analysis_report = agent.run(dataframe=dataframe)
                elif isinstance(agent, ChartPlanningAgent):
                    chart_plan = agent.run(dataframe=dataframe, analysis_report=analysis_report)
                elif isinstance(agent, DataVisualizationAgent):
                    visualization_paths = [
                        str(path)
                        for path in agent.run(
                            dataframe=dataframe,
                            output_dir=output_dir,
                            plan=chart_plan,
                        )
                    ]
                else:
                    agent.run()  # type: ignore[call-arg]

            dataframe_preview = self._render_dataframe_preview(dataframe)
            debug_message = None

        except Exception as exc:  # noqa: BLE001 - centralised error handling
            dataframe_preview = self._render_dataframe_preview(dataframe)
            analysis_report = None
            chart_plan = None
            visualization_paths = []
            debug_message = self.debugger.run(error=exc)

        return OrchestratorResult(
            dataframe_preview=dataframe_preview,
            analysis_report=analysis_report,
            chart_plan=chart_plan,
            visualization_paths=visualization_paths,
            debug_message=debug_message,
        )

    def iter_agents(self) -> Iterable[Agent]:
        yield from self.agents
        yield self.debugger

    @staticmethod
    def _render_dataframe_preview(dataframe: pd.DataFrame | None) -> str:
        if dataframe is None:
            return ""

        head = dataframe.head()
        try:
            return head.to_markdown()
        except ImportError:
            return head.to_string()
        except Exception:
            return head.to_string()
