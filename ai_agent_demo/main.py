"""Command-line entrypoint for the AI agent demo."""
from __future__ import annotations

import argparse

from .agents import (
    ChartPlanningAgent,
    DataAnalysisAgent,
    DataExtractionAgent,
    DataVisualizationAgent,
)
from .orchestrator import AgentOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI agent demo: data extraction, analysis, and visualization",
    )
    parser.add_argument("file", help="Path to the input data file (CSV, TSV, TXT)")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="visualizations",
        help="Directory to save generated charts.",
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help=(
            "Optional OpenAI chat-completions model (e.g. 'gpt-4o-mini') used by the chart-planning agent. "
            "Requires the openai package and OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--llm-temperature",
        dest="llm_temperature",
        type=float,
        default=0.2,
        help="Temperature to use when sampling from the LLM planner (default: 0.2).",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.llm_model:
        chart_planner = ChartPlanningAgent(
            llm_model=parsed.llm_model,
            temperature=parsed.llm_temperature,
        )
        orchestrator = AgentOrchestrator(
            agents=(
                DataExtractionAgent(),
                DataAnalysisAgent(),
                chart_planner,
                DataVisualizationAgent(),
            )
        )
    else:
        orchestrator = AgentOrchestrator()
    result = orchestrator.run(file_path=parsed.file, output_dir=parsed.output_dir)

    print("=== AI Agent Demo ===")
    print("\n-- DataFrame preview --")
    print(result.dataframe_preview)

    if result.analysis_report is not None:
        print("\n-- Analysis report --")
        print(result.analysis_report.to_markdown())

    if result.chart_plan is not None:
        print("\n-- Visualization plan --")
        print(result.chart_plan.to_markdown())

    if result.visualization_paths:
        print("\n-- Generated visualizations --")
        for path in result.visualization_paths:
            print(f"* {path}")

    if result.debug_message:
        print("\n-- Debugging guidance --")
        print(result.debug_message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
