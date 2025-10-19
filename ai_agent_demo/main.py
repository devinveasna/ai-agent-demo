"""Command-line entrypoint for the AI agent demo."""
from __future__ import annotations

import argparse

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
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(file_path=parsed.file, output_dir=parsed.output_dir)

    print("=== AI Agent Demo ===")
    print("\n-- DataFrame preview --")
    print(result.dataframe_preview)

    if result.analysis_report is not None:
        print("\n-- Analysis report --")
        print(result.analysis_report.to_markdown())

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
