"""
Stage 5 — Correlation Filter
Interactive heatmap updates immediately as threshold slider moves.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _corr_pairs(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                rows.append({
                    "Column A": cols[i],
                    "Column B": cols[j],
                    "Correlation": round(val, 4),
                    "|r|": round(abs(val), 4),
                })
    return pd.DataFrame(rows).sort_values("|r|", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def layout(state: dict) -> html.Div:
    if not state.get("df_scaled"):
        return dbc.Alert("Complete Stage 4 — Scaling first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

    # Compute initial correlation
    plot_cols = numeric_cols if len(numeric_cols) <= 80 else \
        X_train[numeric_cols].var().sort_values(ascending=False).head(50).index.tolist()
    corr = X_train[plot_cols].corr()

    init_fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Pearson Correlation Matrix",
    )
    init_fig.update_layout(
        height=max(500, min(len(plot_cols) * 14, 900)),
        margin=dict(t=50, b=20, l=20, r=20),
    )

    return html.Div([
        html.Div([
            html.H2("Stage 5 — Correlation Filter"),
            html.P("Identify and remove highly correlated features."),
        ], className="stage-header"),

        html.P(f"Working with {len(numeric_cols)} numeric columns on the scaled train set.",
               style={"color": "#6b7280"}),

        dbc.Card(dbc.CardBody([
            html.H5("Correlation heatmap"),
            dbc.Row([
                dbc.Col([
                    html.Label("Highlight pairs with |r| >="),
                    dcc.Slider(
                        id="s5-threshold",
                        min=0.5, max=1.0, step=0.05, value=0.8,
                        marks={0.5:"0.5", 0.7:"0.7", 0.8:"0.8", 0.9:"0.9", 1.0:"1.0"},
                    ),
                ], width=8),
            ]),
            dcc.Graph(id="s5-heatmap", figure=init_fig),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Highly correlated pairs"),
            html.Div(id="s5-pairs-table"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Inspect a pair"),
            dcc.Dropdown(id="s5-pair-select", placeholder="Select pair to inspect...", clearable=False),
            dcc.Graph(id="s5-scatter"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Select columns to drop"),
            dcc.Dropdown(
                id="s5-drop-select",
                options=[{"label": c, "value": c} for c in numeric_cols],
                multi=True,
                placeholder="Select columns to remove...",
            ),
            html.Div(id="s5-drop-info", className="mt-2"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Apply filter"),
            dbc.Row([
                dbc.Col(dbc.Button("Remove selected columns", id="s5-apply-btn", color="primary", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("Skip — no columns to drop", id="s5-skip-btn", color="secondary", n_clicks=0), width="auto"),
            ], className="g-2"),
            html.Div(id="s5-apply-status", className="mt-2"),
        ]), className="mb-3"),
    ])


@callback(
    Output("s5-heatmap", "figure"),
    Output("s5-pairs-table", "children"),
    Output("s5-pair-select", "options"),
    Output("s5-drop-select", "value"),
    Input("s5-threshold", "value"),
    prevent_initial_call=True,
)
def update_heatmap_and_pairs(threshold):
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return {}, html.Div(), [], []

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    plot_cols = numeric_cols if len(numeric_cols) <= 80 else \
        X_train[numeric_cols].var().sort_values(ascending=False).head(50).index.tolist()

    corr = X_train[plot_cols].corr()
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=max(500, min(len(plot_cols) * 14, 900)), margin=dict(t=50, b=20, l=20, r=20))

    full_corr = X_train[numeric_cols].corr()
    pairs_df = _corr_pairs(full_corr, threshold or 0.8)

    if pairs_df.empty:
        pairs_tbl = dbc.Alert(f"No pairs exceed |r| = {threshold}.", color="success")
        pair_options = []
        suggested = []
    else:
        pairs_tbl = html.Div([
            dbc.Alert(f"{len(pairs_df)} pairs exceed |r| = {threshold}.", color="warning"),
            dbc.Table.from_dataframe(pairs_df.head(30), striped=True, hover=True, size="sm"),
        ])
        pair_options = [
            {"label": f"{r['Column A']} ↔ {r['Column B']} (r={r['Correlation']})", "value": f"{r['Column A']}||{r['Column B']}"}
            for _, r in pairs_df.iterrows()
        ]
        # Suggest drops
        dropped_set = set()
        suggested = []
        for _, row in pairs_df.iterrows():
            a, b = row["Column A"], row["Column B"]
            if a not in dropped_set and b not in dropped_set:
                suggested.append(b)
                dropped_set.add(b)

    return fig, pairs_tbl, pair_options, suggested


@callback(
    Output("s5-scatter", "figure"),
    Input("s5-pair-select", "value"),
    prevent_initial_call=True,
)
def update_scatter(pair_val):
    if not pair_val:
        return {}
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return {}
    col_a, col_b = pair_val.split("||")
    if col_a not in X_train.columns or col_b not in X_train.columns:
        return {}
    fig = px.scatter(
        x=X_train[col_a], y=X_train[col_b],
        labels={"x": col_a, "y": col_b},
        title=f"Scatter: {col_a} vs {col_b}",
        opacity=0.4,
    )
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    return fig


@callback(
    Output("s5-drop-info", "children"),
    Input("s5-drop-select", "value"),
    prevent_initial_call=True,
)
def drop_info(cols_to_drop):
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return ""
    n_total = len(X_train.columns)
    n_drop = len(cols_to_drop or [])
    return dbc.Alert(
        f"Dropping {n_drop} columns — {n_total - n_drop} will remain.",
        color="info", className="py-2",
    )


@callback(
    Output("s5-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s5-apply-btn", "n_clicks"),
    Input("s5-skip-btn", "n_clicks"),
    State("s5-drop-select", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_filter(n_apply, n_skip, cols_to_drop, state):
    from dash import ctx
    triggered = ctx.triggered_id
    state = dict(state or {})

    if triggered == "s5-skip-btn":
        state["corr_done"] = True
        return dbc.Alert("Skipped correlation filter. Proceed to Stage 6.", color="success"), state

    if not n_apply:
        return no_update, no_update

    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    drops = list(cols_to_drop or [])
    X_tr_new = X_train.drop(columns=drops, errors="ignore")
    X_te_new = X_test.drop(columns=drops, errors="ignore")
    server_store.set_df("X_train", X_tr_new)
    server_store.set_df("X_test", X_te_new)
    state["corr_done"] = True
    state["corr_dropped_cols"] = drops

    return dbc.Alert(
        f"Removed {len(drops)} correlated columns. Remaining features: {X_tr_new.shape[1]}",
        color="success",
    ), state
