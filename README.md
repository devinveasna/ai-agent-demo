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
├── llm.py                      # Shared helpers for configuring OpenAI-compatible clients
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

### Putting LLMs in the loop

Each agent can optionally call an OpenAI-compatible chat endpoint. This works both for hosted services (OpenAI, Azure OpenAI, Anthropic's compatibility layer, etc.) and for local servers that expose the same API surface (LM Studio, llama.cpp's REST mode, Ollama, vLLM, and so on).

1. Install the extra dependency:

   ```bash
   pip install openai
   ```

2. Provide connection details:

   ```bash
   export OPENAI_API_KEY=sk-...              # or pass --llm-api-key
   export LLM_BASE_URL="http://localhost:11434/v1"  # optional, for local servers
   ```

3. Enable the agents you want to augment:

   ```bash
   python -m ai_agent_demo.main \
       path/to/data.csv \
       --output-dir charts/ \
       --planner-llm-model llama3 \
       --analysis-llm-model llama3 \
       --debug-llm-model llama3 \
       --llm-base-url "http://localhost:11434/v1" \
       --llm-api-key "EMPTY"  # many local servers ignore the key but the client requires a value
   ```

   Use `--planner-llm-temperature`, `--analysis-llm-temperature`, and `--debug-llm-temperature` to tune sampling for each agent. `--planner-max-charts` controls how many visualizations the planner may request from the LLM.

Behind the scenes:

- **Planner (`ChartPlanningAgent`)** sends the dataset schema, sample rows, and analysis summary to the LLM and expects a JSON chart plan. If the LLM call fails, the agent reverts to heuristics and surfaces warnings.
- **Analysis (`DataAnalysisAgent`)** still computes deterministic statistics but can ask the LLM for follow-up exploration ideas tailored to the dataset profile.
- **Debugging (`DebuggingAgent`)** can summarise failures and propose remediation steps using the configured model.

All LLM calls are optional. When a client cannot be created (missing dependency, offline server, invalid credentials) the agents log a warning in the CLI output and continue with deterministic fallbacks.

### Running against a remote GPU cluster later

Once you're ready to move from a local workstation to a beefier GPU (e.g., an A100 40G), simply point `--llm-base-url` at the remote deployment and supply its credentials. No code changes are required because the demo already targets the standard OpenAI-compatible chat-completions interface.

## Extending the demo

- Swap deterministic agents with alternative LLM-backed implementations.
- Add more specialized agents (e.g., forecasting, natural-language summaries).
- Introduce a planning agent to decide which analysis/visualization steps to execute.

## License

MIT
