"""Command-line entrypoint for the AI agent demo."""
from __future__ import annotations

import argparse
import os

from .agents import (
    ChartPlanningAgent,
    DataAnalysisAgent,
    DataExtractionAgent,
    DataVisualizationAgent,
    DebuggingAgent,
)
from .llm import LLMSettings
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
        "--planner-llm-model",
        dest="planner_llm_model",
        default=None,
        help=(
            "Name of an OpenAI-compatible chat-completions model used by the chart-planning agent. "
            "Works with hosted endpoints or local servers that expose the OpenAI API surface."
        ),
    )
    parser.add_argument(
        "--llm-model",
        dest="planner_llm_model",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--planner-llm-temperature",
        dest="planner_llm_temperature",
        type=float,
        default=0.2,
        help="Temperature to use when sampling chart plans from the LLM (default: 0.2).",
    )
    parser.add_argument(
        "--planner-max-charts",
        dest="planner_max_charts",
        type=int,
        default=6,
        help="Maximum number of charts to request from the planner LLM (default: 6).",
    )
    parser.add_argument(
        "--analysis-llm-model",
        dest="analysis_llm_model",
        default=None,
        help="Optional OpenAI-compatible model used to expand the analysis recommendations.",
    )
    parser.add_argument(
        "--analysis-llm-temperature",
        dest="analysis_llm_temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the analysis agent when it calls an LLM (default: 0.2).",
    )
    parser.add_argument(
        "--debug-llm-model",
        dest="debug_llm_model",
        default=None,
        help="Optional OpenAI-compatible model used to enrich debugging tips.",
    )
    parser.add_argument(
        "--debug-llm-temperature",
        dest="debug_llm_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the debugging agent when it calls an LLM (default: 0.0).",
    )
    parser.add_argument(
        "--llm-base-url",
        dest="llm_base_url",
        default=None,
        help=(
            "Base URL for the OpenAI-compatible endpoint. Set this to your local server (e.g. "
            "'http://localhost:11434/v1') when running an on-prem LLM."
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        dest="llm_api_key",
        default=None,
        help="API key used to authenticate with the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-timeout",
        dest="llm_timeout",
        type=float,
        default=None,
        help="Optional request timeout (seconds) for LLM API calls.",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    llm_base_url = parsed.llm_base_url or os.getenv("LLM_BASE_URL")
    llm_api_key = parsed.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if parsed.llm_timeout is not None:
        llm_timeout = parsed.llm_timeout
    else:
        timeout_env = os.getenv("LLM_TIMEOUT")
        llm_timeout = float(timeout_env) if timeout_env else None

    def build_settings(model: str | None) -> LLMSettings | None:
        if not model:
            return None
        return LLMSettings(
            model=model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout=llm_timeout,
        )

    analysis_agent = DataAnalysisAgent(
        llm_settings=build_settings(parsed.analysis_llm_model),
        llm_temperature=parsed.analysis_llm_temperature,
    )
    chart_planner = ChartPlanningAgent(
        llm_settings=build_settings(parsed.planner_llm_model),
        temperature=parsed.planner_llm_temperature,
        max_charts=parsed.planner_max_charts,
    )
    debugger = DebuggingAgent(
        llm_settings=build_settings(parsed.debug_llm_model),
        llm_temperature=parsed.debug_llm_temperature,
    )

    orchestrator = AgentOrchestrator(
        agents=(
            DataExtractionAgent(),
            analysis_agent,
            chart_planner,
            DataVisualizationAgent(),
        ),
        debugger=debugger,
    )
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
