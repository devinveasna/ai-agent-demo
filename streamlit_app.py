"""Streamlit user interface for the AI agent demo."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from ai_agent_demo.agents.data_visualization import VisualizationRequest
from ai_agent_demo.orchestrator import AgentOrchestrator


st.set_page_config(page_title="AI Agent Demo", layout="wide")
st.title("AI Agent Demo: Visualization Planner")
st.write(
    "Upload a dataset and tell the agents what kind of charts you would like to see."
)


@st.cache_data(show_spinner=False)
def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _load_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, Path]:
    suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)

    dataframe = _load_dataframe(temp_path)
    return dataframe, temp_path


def _load_sample_dataset() -> tuple[pd.DataFrame, Path]:
    sample_path = Path("sample.csv")
    dataframe = _load_dataframe(sample_path)
    return dataframe, sample_path


def build_visualization_requests(
    *,
    histogram_columns: List[str],
    scatter_pair: tuple[str, str] | None,
) -> List[VisualizationRequest]:
    requests: List[VisualizationRequest] = []
    for column in histogram_columns:
        requests.append(VisualizationRequest(chart_type="histogram", columns=[column]))

    if scatter_pair and scatter_pair[0] != scatter_pair[1]:
        requests.append(
            VisualizationRequest(chart_type="scatter", columns=[scatter_pair[0], scatter_pair[1]])
        )

    return requests


uploaded_file = st.file_uploader(
    "Upload a CSV, TSV, or TXT file", type=["csv", "tsv", "txt"], help="Select your dataset."
)

use_sample = st.checkbox("Use bundled sample dataset instead")

dataframe: pd.DataFrame | None = None
data_path: Path | None = None

if uploaded_file is not None and not use_sample:
    try:
        dataframe, data_path = _load_uploaded_file(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - surfacing errors to the UI
        st.error(f"Failed to load the uploaded file: {exc}")
elif use_sample:
    try:
        dataframe, data_path = _load_sample_dataset()
    except Exception as exc:  # noqa: BLE001 - surfacing errors to the UI
        st.error(f"Failed to load the sample dataset: {exc}")


if dataframe is None:
    st.info("Upload a dataset or enable the sample dataset to get started.")
    st.stop()


st.subheader("Dataset preview")
st.dataframe(dataframe.head(), use_container_width=True)

numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

if not numeric_columns:
    st.warning("No numeric columns detected. Visualization options are limited.")
    st.stop()


st.sidebar.header("Visualization preferences")
st.sidebar.write("Choose the charts you want the visualization agent to create.")

default_hist_columns = numeric_columns[: min(3, len(numeric_columns))]
histogram_columns = st.sidebar.multiselect(
    "Columns for histograms",
    options=numeric_columns,
    default=default_hist_columns,
    help="Histograms help inspect the distribution of individual numeric columns.",
)

scatter_pair: tuple[str, str] | None = None
if len(numeric_columns) >= 2:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scatter plot")
    scatter_enabled = st.sidebar.checkbox(
        "Generate a scatter plot",
        value=len(numeric_columns) >= 2,
        help="Scatter plots show relationships between two numeric columns.",
    )
    if scatter_enabled:
        x_column = st.sidebar.selectbox("X-axis", numeric_columns, index=0)
        y_options = [col for col in numeric_columns if col != x_column]
        if y_options:
            y_column = st.sidebar.selectbox("Y-axis", y_options, index=0)
            scatter_pair = (x_column, y_column)
        else:
            st.sidebar.info("Select a different X-axis column to enable the Y-axis selection.")
else:
    st.sidebar.info("At least two numeric columns are required for scatter plots.")


output_directory = st.sidebar.text_input(
    "Output directory",
    value="visualizations",
    help="Generated charts will be saved in this directory.",
)


run_pipeline = st.sidebar.button("Run agents")

if not run_pipeline:
    st.stop()


visualization_requests = build_visualization_requests(
    histogram_columns=histogram_columns,
    scatter_pair=scatter_pair,
)

orchestrator = AgentOrchestrator()
result = orchestrator.run(
    file_path=str(data_path),
    output_dir=output_directory,
    visualization_requests=visualization_requests,
)

st.success("Agents finished running!")

st.subheader("Analysis report")
if result.analysis_report is not None:
    st.markdown(result.analysis_report.to_markdown())
else:
    st.write("No analysis report was generated.")


st.subheader("Generated visualizations")
if result.visualization_paths:
    for path_str in result.visualization_paths:
        st.markdown(f"**{path_str}**")
        st.image(path_str, use_column_width=True)
else:
    st.write("No visualizations were created. Adjust your selections and try again.")


if result.debug_message:
    st.subheader("Debugging guidance")
    st.warning(result.debug_message)


if data_path and data_path.is_file() and data_path.name.startswith("tmp"):
    try:
        data_path.unlink(missing_ok=True)
    except OSError:
        pass
