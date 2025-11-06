# AI Agent Demo

This project demonstrates a lightweight, multi-agent workflow for turning local tabular data files into visual insights. While the agents are implemented with deterministic Python logic, the architecture mirrors how you might coordinate multiple LLM-powered workers in a production system.

## Project structure

```
ai_agent_demo/
├── agents/
│   ├── base.py                 # Common agent abstractions and errors
│   ├── data_extraction.py      # Loads CSV/TSV/TXT files into pandas DataFrames
│   ├── data_analysis.py        # Generates descriptive statistics and suggestions
│   ├── data_visualization.py   # Creates histograms and scatter plots with matplotlib
│   └── debugging.py            # Produces human-friendly debugging tips
├── orchestrator.py             # Coordinates the agents into a pipeline
└── main.py                     # CLI entrypoint for running the demo
```

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the CLI demo**

   ```bash
   python -m ai_agent_demo.main path/to/data.csv --output-dir charts/
   ```

   The command prints a DataFrame preview, a Markdown-formatted analysis report, and a list of generated visualization files. Histograms are always produced for each numeric column, and a scatter plot is created for the first two numeric columns when available.

   > **Note:** Pandas relies on the optional [`tabulate`](https://pypi.org/project/tabulate/) package to render the preview in Markdown. If `tabulate` is not installed, the orchestrator falls back to a plain-text table so the demo continues to run without extra dependencies.

3. **Launch the Streamlit UI (optional)**

   A simple Streamlit front-end is provided for exploring the agents without using the command line.

   ```bash
   streamlit run streamlit_app.py
   ```

   From the Streamlit dashboard you can upload your own CSV/TSV/TXT file or toggle the bundled `sample.csv`, choose which histograms and scatter plots to generate, and let the orchestrator display the resulting analysis and charts directly in the browser. Visualizations are saved to the directory you specify in the sidebar so that you can download or reuse them later.

## Extending the demo

- Swap deterministic agents with LLM-backed implementations.
- Add more specialized agents (e.g., forecasting, natural-language summaries).
- Introduce a planning agent to decide which analysis/visualization steps to execute.

## License

MIT
