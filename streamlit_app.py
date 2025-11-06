"""Streamlit user interface for the AI agent demo."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image

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


MAX_INLINE_IMAGE_WIDTH = 800


def build_visualization_requests(
    *,
    histogram_columns: List[str],
    scatter_pairs: List[tuple[str, str]],
) -> List[VisualizationRequest]:
    requests: List[VisualizationRequest] = []
    for column in histogram_columns:
        requests.append(VisualizationRequest(chart_type="histogram", columns=[column]))

    for x_column, y_column in scatter_pairs:
        if x_column == y_column:
            continue
        requests.append(
            VisualizationRequest(chart_type="scatter", columns=[x_column, y_column])
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

chart_type_options: list[tuple[str, str]] = []
if numeric_columns:
    chart_type_options.append(("Histograms", "histogram"))
if len(numeric_columns) >= 2:
    chart_type_options.append(("Scatter plots", "scatter"))

if not chart_type_options:
    st.sidebar.info("No supported chart types available for the detected data.")
    st.stop()

default_selected_labels = [label for label, _ in chart_type_options]
selected_chart_labels = st.sidebar.multiselect(
    "Visualization types", [label for label, _ in chart_type_options], default=default_selected_labels
)

selected_chart_types = {value for label, value in chart_type_options if label in selected_chart_labels}

histogram_columns: list[str] = []
if "histogram" in selected_chart_types:
    default_hist_columns = numeric_columns[: min(3, len(numeric_columns))]
    histogram_columns = st.sidebar.multiselect(
        "Columns for histograms",
        options=numeric_columns,
        default=default_hist_columns,
        help="Histograms help inspect the distribution of individual numeric columns.",
    )

scatter_pairs: list[tuple[str, str]] = []
if "scatter" in selected_chart_types:
    if len(numeric_columns) < 2:
        st.sidebar.info("At least two numeric columns are required for scatter plots.")
    else:
        from itertools import permutations

        scatter_pair_options = [
            (f"{x} vs {y}", (x, y)) for x, y in permutations(numeric_columns, 2) if x != y
        ]
        if scatter_pair_options:
            default_option_label = scatter_pair_options[0][0]
            selected_pairs = st.sidebar.multiselect(
                "Scatter plot combinations",
                options=[label for label, _ in scatter_pair_options],
                default=[default_option_label],
                help="Select the combinations of numeric columns to compare in scatter plots.",
            )
            lookup = {label: pair for label, pair in scatter_pair_options}
            scatter_pairs = [lookup[label] for label in selected_pairs if label in lookup]
        else:
            st.sidebar.info(
                "No unique scatter plot combinations available. Select additional numeric columns."
            )

if not histogram_columns and not scatter_pairs:
    st.info("Select at least one visualization option to run the agents.")
    st.stop()


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
    scatter_pairs=scatter_pairs,
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
        image_path = Path(path_str)
        try:
            with Image.open(image_path) as loaded_image:
                image = loaded_image.copy()
        except (FileNotFoundError, OSError) as exc:
            st.warning(f"Unable to display visualization '{path_str}': {exc}")
            continue

        use_container_width = image.width > MAX_INLINE_IMAGE_WIDTH
        display_width = None if use_container_width else image.width
        st.image(
            image,
            use_container_width=use_container_width,
            width=display_width,
        )
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
