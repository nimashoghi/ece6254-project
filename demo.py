from __future__ import annotations

import os
from typing import Any

import crystal_toolkit.components as ctc
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dash import dcc, html
from dash.dependencies import Input, Output, State
from pymatgen.core import Structure

pio.templates.default = "presentation"

df = pd.read_pickle(os.environ.get("INPUT_FILE", "./tsne-final.pkl"))

subsample_n_per_model = None
if subsample_n_per_model is not None:
    df = (
        df.groupby("model")
        .apply(lambda x: x.sample(subsample_n_per_model))
        .reset_index(drop=True)
    )


model_name_mapping = {
    "flowmm": "FlowMM",
    "cdvae": "CDVAE",
    "crystal-text-llm": "CrystalTextLLM",
    "ground_truth_with_embeddings": "Ground Truth",
}
df["model"] = df["model"].map(model_name_mapping)

plot_labels = {
    "model": "Model",
}
colors = {
    "FlowMM": "rgba(188, 128, 189, 0.75)",
    "CDVAE": "rgba(255, 127, 0, 0.75)",
    "CrystalTextLLM": "rgba(77, 175, 74, 0.75)",
    "Ground Truth": "rgba(55, 126, 184, 1.0)",
}
fig = px.scatter(
    df,
    x="2d_x",
    y="2d_y",
    color="model",
    color_discrete_map=colors,
    labels=plot_labels,
    hover_name="dataset_full_name",
    # hover_data=["subset"],
    # size_max=5,  # Control the maximum marker size
)
# Set marker size directly
for trace in fig.data:
    trace.marker.size = 4  # Smaller dot size
    # Make all models except Ground Truth slightly transparent

title = "Generated Materials, UMAP of Embeddings"
fig.update_layout(
    title=dict(text=f"<b>{title}</b>", x=0.5, font_size=20),
    margin=dict(b=20, l=40, r=20, t=100),
)
fig.update_layout(showlegend=True)

fig.update_xaxes(title_text="", showticklabels=False)
fig.update_yaxes(title_text="", showticklabels=False)

# Hide the gridlines
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)


SELECTED_DICT = {
    "marker": {
        "color": "red",
        "size": 8,
        "opacity": 0.95,
    },
}


structure_component = ctc.StructureMoleculeComponent(
    None,
    id="structure",
)
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body style="background-color: white !important;">
        <!--[if IE]><script>
        alert("Dash v2.7+ does not support Internet Explorer. Please use a newer browser.");
        </script><![endif]-->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
graph = dcc.Graph(
    id="tsne-scatter-plot",
    figure=fig,
    style={"width": "100%"},
)
struct_title = html.H2(
    "Try clicking on a point in the plot to see its corresponding structure here",
    id="struct-title",
    style=dict(position="absolute", padding="1ex 1em", maxWidth="25em"),
)
app.layout = dbc.Row(
    [
        dbc.Col([graph], md=6, sm=12),
        dbc.Col([struct_title, structure_component.layout(size="100%")], md=6, sm=12),
    ],
)
ctc.register_crystal_toolkit(app=app, layout=app.layout)


@app.callback(
    Output(structure_component.id(), "data"),
    Output(struct_title, "children"),
    Output(graph, "figure"),
    Input(graph, "clickData"),
    State(graph, "figure"),
)
def update_structure(
    click_data: dict[str, list[dict[str, Any]]],
    current_fig: dict,
) -> tuple[Structure, str, dict]:
    if (
        click_data is None
        or (points := click_data.get("points")) is None
        or len(points) == 0
    ):
        raise dash.exceptions.PreventUpdate

    data = click_data["points"][0]

    curve_number = data.get("curveNumber", 0)
    unique_dataset_list = list(df["model"].unique())
    df_filtered = df[df["model"] == unique_dataset_list[curve_number]]

    point_idx = data.get("pointIndex", 0)
    row = df_filtered.iloc[point_idx]

    structure = row.structure
    dataset = row.dataset_full_name

    # Update the style of the selected point in the t-SNE plot
    updated_fig = current_fig.copy()
    selected_point = SELECTED_DICT
    # Unselect the previous points
    for data in updated_fig["data"]:
        data.pop("selectedpoints", None)
        data.pop("selected", None)

    updated_fig["data"][curve_number]["selectedpoints"] = [point_idx]
    updated_fig["data"][curve_number]["selected"] = selected_point

    return structure, dataset, updated_fig


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
