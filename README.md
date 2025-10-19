# AI Agent Demo

This project demonstrates a lightweight, multi-agent workflow for turning local tabular data files into visual insights. While the agents are implemented with deterministic Python logic, the architecture mirrors how you might coordinate multiple LLM-powered workers in a production system.

## Project structure

```
ai_agent_demo/
├── agents/
│   ├── base.py                 # Common agent abstractions and errors
│   ├── data_extraction.py      # Loads CSV/TSV/TXT files into pandas DataFrames
│   ├── data_analysis.py        # Generates descriptive statistics and suggestions
│   ├── chart_planning.py       # Suggests charts via heuristics or an optional LLM
│   ├── data_visualization.py   # Builds charts from the planner's recommendations
│   └── debugging.py            # Produces human-friendly debugging tips
├── orchestrator.py             # Coordinates the agents into a pipeline
└── main.py                     # CLI entrypoint for running the demo
```

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**

   ```bash
   python -m ai_agent_demo.main path/to/data.csv --output-dir charts/
   ```

   The command prints a DataFrame preview, a Markdown-formatted analysis report, a visualization plan, and a list of generated charts. By default the planner uses heuristics to suggest histograms for numeric columns, scatter plots for the first two numeric columns, and a bar chart for the most prominent categorical feature.

   > **Note:** Pandas relies on the optional [`tabulate`](https://pypi.org/project/tabulate/) package to render the preview in Markdown. If `tabulate` is not installed, the orchestrator falls back to a plain-text table so the demo continues to run without extra dependencies.

### Using an LLM to plan charts

The planner can delegate chart selection to an OpenAI-compatible chat model. Install the optional dependency and provide an API key:

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python -m ai_agent_demo.main path/to/data.csv --llm-model gpt-4o-mini
```

Additional options:

- `--llm-temperature` adjusts how adventurous the LLM is when proposing chart types (default: `0.2`).
- Planner warnings (for example missing columns suggested by the LLM) are surfaced beneath the plan in the CLI output.

## Extending the demo

- Swap deterministic agents with alternative LLM-backed implementations.
- Add more specialized agents (e.g., forecasting, natural-language summaries).
- Introduce a planning agent to decide which analysis/visualization steps to execute.

## License

MIT
