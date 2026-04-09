"""
Stage 2.5 — Visualisation Dashboard
Parse questionnaire, build codebook, render interactive chart dashboard.
"""
import base64, io, tempfile, os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store
from utils.qnr_parser import parse_questionnaire
from utils.col_mapper import build_codebook, group_columns

CHART_OPTIONS = {
    "categorical": ["Bar chart", "Pie chart", "Donut chart"],
    "ordinal":     ["Bar chart", "Horizontal bar", "Stacked bar", "Line chart"],
    "numeric":     ["Histogram", "Box plot", "Violin plot", "Line chart"],
    "scale_7":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "scale_5":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "multi":       ["Bar chart", "Horizontal bar"],
    "grid":        ["Heatmap", "Grouped bar"],
    "open":        ["Word count bar"],
    "single":      ["Bar chart", "Pie chart"],
}
VAR_TYPES = ["categorical", "ordinal", "numeric", "scale_7", "scale_5", "multi", "grid", "open", "single"]


def _render_chart(series: pd.Series, tile: dict, df: pd.DataFrame) -> go.Figure:
    chart = tile.get("chart_type", "Bar chart")
    title = tile.get("code", "")
    labels = tile.get("value_labels", {})
    if labels:
        series = series.map(lambda x: labels.get(str(x), x))
    counts = series.value_counts().sort_index()

    if chart in ("Bar chart", "Single"):
        fig = px.bar(x=counts.index.astype(str), y=counts.values,
                     labels={"x": "", "y": "Count"}, title=title)
        fig.update_layout(showlegend=False)
    elif chart == "Horizontal bar":
        fig = px.bar(x=counts.values, y=counts.index.astype(str), orientation="h",
                     labels={"x": "Count", "y": ""}, title=title)
        fig.update_layout(yaxis=dict(autorange="reversed"))
    elif chart == "Pie chart":
        fig = px.pie(values=counts.values, names=counts.index.astype(str), title=title)
    elif chart == "Donut chart":
        fig = px.pie(values=counts.values, names=counts.index.astype(str), hole=0.4, title=title)
    elif chart == "Histogram":
        fig = px.histogram(series.dropna(), title=title, nbins=20)
    elif chart == "Box plot":
        fig = px.box(y=series.dropna(), title=title)
    elif chart == "Violin plot":
        fig = px.violin(y=series.dropna(), title=title, box=True)
    elif chart == "Line chart":
        fig = px.line(x=counts.index.astype(str), y=counts.values, markers=True, title=title)
    elif chart == "Mean line":
        vals = pd.to_numeric(series, errors="coerce").dropna()
        mean_val = vals.mean()
        fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"{title} (mean={mean_val:.2f})")
        fig.add_hline(y=mean_val, line_dash="dash", line_color="red")
    elif chart == "Word count bar":
        words = series.dropna().astype(str).str.lower().str.split().explode()
        wc = words.value_counts().head(20)
        fig = px.bar(x=wc.values, y=wc.index, orientation="h", title=f"{title} — top words")
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = px.bar(x=counts.index.astype(str), y=counts.values, title=title)

    fig.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20), font=dict(size=10))
    return fig


def _get_viz_df(state):
    X_train = server_store.get_df("X_train")
    if X_train is not None:
        target = state.get("target_col", "")
        df = X_train.copy()
        y = server_store.get_df("y_train")
        if y is not None and target:
            df[target] = y.values
        return df
    return server_store.get_df("viz_df")


def layout(state: dict) -> html.Div:
    df = _get_viz_df(state)
    has_data = df is not None
    codebook = server_store.get_val("codebook")

    mapped = []
    if codebook and df is not None:
        mapped = [t for t in codebook if t.get("include") and t.get("dataset_col") and t["dataset_col"] in df.columns]

    # Build grid of charts
    charts_grid = html.Div()
    if mapped and df is not None:
        tiles_per_row = server_store.get_val("viz_tiles_per_row") or 3
        chart_cols = []
        row_tiles = []
        for tile in mapped:
            col = tile.get("dataset_col")
            if not col or col not in df.columns:
                continue
            try:
                fig = _render_chart(df[col], tile, df)
                card = dbc.Card(dbc.CardBody([
                    html.Small(f"{tile['code']} — {tile['question'][:50]}",
                               style={"color": "#6b7280"}),
                    dcc.Graph(figure=fig, style={"height": "290px"}),
                ]), className="mb-2")
            except Exception as e:
                card = dbc.Alert(f"Chart error ({col}): {e}", color="danger", className="mb-2")
            row_tiles.append(dbc.Col(card, width=12 // min(tiles_per_row, 3)))
            if len(row_tiles) >= tiles_per_row:
                chart_cols.append(dbc.Row(row_tiles, className="mb-2"))
                row_tiles = []
        if row_tiles:
            chart_cols.append(dbc.Row(row_tiles))
        charts_grid = html.Div(chart_cols)

    return html.Div([
        html.Div([
            html.H2("Stage 2.5 — Visualisation Dashboard"),
            html.P("Parse questionnaire, map variables, build interactive dashboard."),
        ], className="stage-header"),

        # Step 1: Questionnaire upload
        dbc.Card(dbc.CardBody([
            html.H5("Step 1 — Upload questionnaire document (optional)"),
            dcc.Upload(
                id="s2b-qnr-upload",
                children=html.Div([html.I(className="bi bi-file-text me-2"), "Upload Word/PDF/Excel/TXT"]),
                accept=".docx,.pdf,.xlsx,.xls,.txt",
                className="upload-area",
            ),
            html.Div(id="s2b-qnr-status"),
        ]), className="mb-3"),

        # Step 2: Dataset
        dbc.Card(dbc.CardBody([
            html.H5("Step 2 — Dataset"),
            html.Div(
                dbc.Alert(f"Using dataset from Stage 0/1 — {df.shape[0]} rows, {df.shape[1]} columns.", color="success")
                if has_data else
                html.Div([
                    dcc.Upload(
                        id="s2b-data-upload",
                        children=html.Div([html.I(className="bi bi-cloud-upload me-2"), "Upload CSV/Excel"]),
                        accept=".csv,.xlsx,.xls",
                        className="upload-area",
                    ),
                    html.Div(id="s2b-data-status"),
                ])
            ),
        ]), className="mb-3"),

        # Step 3: Codebook
        dbc.Card(dbc.CardBody([
            html.H5("Step 3 — Build codebook"),
            dbc.Button("Build / Rebuild codebook from data", id="s2b-build-btn", color="secondary", n_clicks=0),
            html.Div(id="s2b-build-status", className="mt-2"),
            html.Div(
                html.Div(
                    f"{len(mapped)} questions mapped." if codebook else "No codebook yet — click Build.",
                    style={"color": "#6b7280", "fontSize": "0.85rem", "marginTop": "8px"},
                )
            ),
        ]), className="mb-3"),

        # Step 4: Dashboard controls
        dbc.Card(dbc.CardBody([
            html.H5("Step 4 — Dashboard"),
            dbc.Row([
                dbc.Col([
                    html.Label("Tiles per row"),
                    dcc.Slider(id="s2b-tiles-per-row", min=1, max=4, step=1, value=3,
                               marks={1:"1", 2:"2", 3:"3", 4:"4"}),
                ], width=4),
            ]),
            html.Div(id="s2b-charts-grid", children=charts_grid),
        ]), className="mb-3"),
    ])


@callback(
    Output("s2b-qnr-status", "children"),
    Input("s2b-qnr-upload", "contents"),
    State("s2b-qnr-upload", "filename"),
    prevent_initial_call=True,
)
def handle_qnr_upload(contents, filename):
    if not contents:
        return no_update
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        parsed = parse_questionnaire(tmp_path)
        os.unlink(tmp_path)
        server_store.set_val("parsed_questions", parsed)
        return dbc.Alert(f"Parsed {len(parsed)} questions from '{filename}'", color="success")
    except Exception as e:
        return dbc.Alert(f"Parse error: {e}", color="danger")


@callback(
    Output("s2b-data-status", "children"),
    Input("s2b-data-upload", "contents"),
    State("s2b-data-upload", "filename"),
    prevent_initial_call=True,
)
def handle_data_upload(contents, filename):
    if not contents:
        return no_update
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
        else:
            df = pd.read_excel(io.BytesIO(decoded))
        server_store.set_df("viz_df", df)
        return dbc.Alert(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns", color="success")
    except Exception as e:
        return dbc.Alert(f"Upload error: {e}", color="danger")


@callback(
    Output("s2b-build-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s2b-build-btn", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def build_cb(n_clicks, state):
    if not n_clicks:
        return no_update, no_update

    state = dict(state or {})
    X_train = server_store.get_df("X_train")
    if X_train is not None:
        target = state.get("target_col", "")
        df = X_train.copy()
        y = server_store.get_df("y_train")
        if y is not None and target:
            df[target] = y.values
    else:
        df = server_store.get_df("viz_df")

    if df is None:
        return dbc.Alert("No dataset available.", color="warning"), no_update

    parsed = server_store.get_val("parsed_questions") or []
    if parsed:
        codebook = build_codebook(df, parsed)
    else:
        col_groups = group_columns(df)
        codebook = []
        for prefix, grp in col_groups.items():
            vt = "numeric" if pd.api.types.is_numeric_dtype(df[grp["primary"]]) else "categorical"
            codebook.append({
                "code": prefix, "question": prefix, "var_type": vt,
                "chart_type": CHART_OPTIONS.get(vt, ["Bar chart"])[0],
                "group_type": grp["group_type"],
                "dataset_col": grp["primary"], "all_cols": grp["cols"],
                "options": [], "value_labels": {}, "scale_low": "", "scale_high": "",
                "scale_points": None, "pn_notes": "", "include": True,
            })

    for t in codebook:
        if "options" not in t:
            t["options"] = []

    server_store.set_val("codebook", codebook)
    n_mapped = sum(1 for t in codebook if t.get("include") and t.get("dataset_col") and t["dataset_col"] in df.columns)
    state["codebook_ready"] = True

    return dbc.Alert(f"Codebook built — {n_mapped} questions mapped to dataset columns.", color="success"), state


@callback(
    Output("s2b-charts-grid", "children"),
    Input("s2b-tiles-per-row", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def update_tiles(tiles_per_row, state):
    tiles_per_row = tiles_per_row or 3
    server_store.set_val("viz_tiles_per_row", tiles_per_row)
    codebook = server_store.get_val("codebook")
    df = _get_viz_df(state or {})
    if not codebook or df is None:
        return dbc.Alert("Build codebook first.", color="info")

    mapped = [t for t in codebook if t.get("include") and t.get("dataset_col") and t["dataset_col"] in df.columns]
    if not mapped:
        return dbc.Alert("No mapped questions.", color="info")

    chart_rows = []
    row_tiles = []
    for tile in mapped:
        col = tile.get("dataset_col")
        try:
            fig = _render_chart(df[col], tile, df)
            card = dbc.Card(dbc.CardBody([
                html.Small(f"{tile['code']} — {tile['question'][:50]}", style={"color": "#6b7280"}),
                dcc.Graph(figure=fig, style={"height": "290px"}),
            ]), className="mb-2")
        except Exception as e:
            card = dbc.Alert(f"Chart error ({col}): {e}", color="danger")
        row_tiles.append(dbc.Col(card, width=12 // min(tiles_per_row, 4)))
        if len(row_tiles) >= tiles_per_row:
            chart_rows.append(dbc.Row(row_tiles, className="mb-2"))
            row_tiles = []
    if row_tiles:
        chart_rows.append(dbc.Row(row_tiles))
    return html.Div(chart_rows)
