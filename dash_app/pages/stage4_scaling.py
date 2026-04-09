"""
Stage 4 — Feature Scaling
Scaler fitted on X_train only, then applied to both splits.
"""
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


SCALER_MAP = {
    "RobustScaler (recommended — handles outliers)": RobustScaler,
    "StandardScaler (zero mean, unit std)": StandardScaler,
    "MinMaxScaler ([0, 1] range)": MinMaxScaler,
    "None (passthrough — no scaling)": None,
}


def _is_binary(series):
    uv = set(series.dropna().unique())
    return uv <= {0, 1}


def layout(state: dict) -> html.Div:
    if not state.get("df_encoded"):
        return dbc.Alert("Complete Stage 3 — Encoding first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    all_cols = list(X_train.columns)
    numeric_scalable = [c for c in all_cols if pd.api.types.is_numeric_dtype(X_train[c]) and not _is_binary(X_train[c])]
    binary_cols = [c for c in all_cols if _is_binary(X_train[c])]
    non_numeric = [c for c in all_cols if not pd.api.types.is_numeric_dtype(X_train[c])]

    # Preview histogram for first numeric col
    preview_options = [{"label": c, "value": c} for c in numeric_scalable] if numeric_scalable else []

    return html.Div([
        html.Div([
            html.H2("Stage 4 — Scaling"),
            html.P("Scale numeric features. Scaler fitted on train set only to prevent leakage."),
        ], className="stage-header"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Scalable numeric", className="metric-label"), html.Div(len(numeric_scalable), className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Binary (skip)", className="metric-label"), html.Div(len(binary_cols), className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Non-numeric (warn)", className="metric-label"), html.Div(len(non_numeric), className="metric-value", style={"color": "#ef4444" if non_numeric else "inherit"})]), className="metric-card"), width=3),
        ], className="mb-3"),

        dbc.Alert(
            f"Non-numeric columns detected — encode them in Stage 3: {', '.join(non_numeric[:8])}",
            color="warning",
        ) if non_numeric else html.Div(),

        dbc.Card(dbc.CardBody([
            html.H5("Choose scaler"),
            dcc.Dropdown(
                id="s4-scaler-choice",
                options=[{"label": k, "value": k} for k in SCALER_MAP.keys()],
                value="RobustScaler (recommended — handles outliers)",
                clearable=False,
            ),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Exclude columns from scaling (optional)"),
            dcc.Dropdown(
                id="s4-exclude",
                options=[{"label": c, "value": c} for c in numeric_scalable],
                multi=True,
                placeholder="Select columns to exclude from scaling...",
            ),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Distribution preview (before scaling)"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="s4-preview-col",
                    options=preview_options,
                    value=numeric_scalable[0] if numeric_scalable else None,
                    clearable=False,
                ), width=4),
            ]),
            dcc.Graph(id="s4-preview-hist"),
        ]), className="mb-3") if numeric_scalable else html.Div(),

        dbc.Card(dbc.CardBody([
            html.H5("Apply scaling"),
            dbc.Button("Apply scaling", id="s4-apply-btn", color="primary", n_clicks=0),
            html.Div(id="s4-apply-status", className="mt-2"),
        ]), className="mb-3"),
    ])


@callback(
    Output("s4-preview-hist", "figure"),
    Input("s4-preview-col", "value"),
    prevent_initial_call=True,
)
def preview_histogram(col):
    X_train = server_store.get_df("X_train")
    if X_train is None or not col or col not in X_train.columns:
        return {}
    fig = px.histogram(X_train[col].dropna(), nbins=30,
                       title=f"{col} — distribution before scaling",
                       labels={"value": col})
    fig.update_layout(height=260, margin=dict(t=40, b=20))
    return fig


@callback(
    Output("s4-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s4-apply-btn", "n_clicks"),
    State("s4-scaler-choice", "value"),
    State("s4-exclude", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_scaling(n_clicks, scaler_choice, exclude_cols, state):
    if not n_clicks:
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    all_cols = list(X_train.columns)
    numeric_scalable = [c for c in all_cols if pd.api.types.is_numeric_dtype(X_train[c]) and not _is_binary(X_train[c])]
    exclude = set(exclude_cols or [])

    X_tr = X_train.copy()
    X_te = X_test.copy()
    global_cls = SCALER_MAP.get(scaler_choice or "")
    scalers_used = {}

    for col in numeric_scalable:
        if col in exclude:
            continue
        if global_cls is None:
            continue
        scaler = global_cls()
        X_tr[[col]] = scaler.fit_transform(X_tr[[col]])
        X_te[[col]] = scaler.transform(X_te[[col]])
        scalers_used[col] = scaler

    server_store.set_df("X_train", X_tr)
    server_store.set_df("X_test", X_te)
    server_store.set_val("scalers_used", scalers_used)

    state = dict(state or {})
    state["df_scaled"] = True

    # Stats table
    if scalers_used:
        stats = X_tr[list(scalers_used.keys())].describe().T[["mean", "std", "min", "max"]].round(3)
        stats_table = dbc.Table.from_dataframe(stats.reset_index().rename(columns={"index": "Column"}),
                                               striped=True, hover=True, size="sm")
    else:
        stats_table = html.Div()

    return html.Div([
        dbc.Alert(
            f"Scaling applied to {len(scalers_used)} columns. "
            f"Train: {X_tr.shape[0]}×{X_tr.shape[1]}  Test: {X_te.shape[0]}×{X_te.shape[1]}",
            color="success",
        ),
        html.H6("Scaled column statistics:"),
        stats_table,
    ]), state
