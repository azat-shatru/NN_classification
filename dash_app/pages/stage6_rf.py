"""
Stage 6 — Random Forest Feature Importance
Train RF, rank features, top-N slider updates chart immediately.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def layout(state: dict) -> html.Div:
    if not state.get("corr_done"):
        return dbc.Alert("Complete Stage 5 — Correlation Filter first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    importances = server_store.get_val("rf_importances")

    # Build importance chart if available
    imp_section = html.Div()
    if importances is not None:
        n_feat = len(importances)
        top_n_init = min(30, n_feat)
        imp_section = html.Div([
            dbc.Card(dbc.CardBody([
                html.H5("Feature importance chart"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Show top N features"),
                        dcc.Slider(
                            id="s6-topn",
                            min=5, max=n_feat, step=1, value=top_n_init,
                            marks={5:"5", n_feat//2: str(n_feat//2), n_feat: str(n_feat)},
                        ),
                    ], width=8),
                ]),
                dcc.Graph(id="s6-imp-chart"),
            ]), className="mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Full importance table"),
                html.Div(
                    style={"maxHeight": "300px", "overflowY": "auto"},
                    children=dbc.Table.from_dataframe(importances, striped=True, hover=True, size="sm"),
                ),
            ]), className="mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Select features to keep"),
                dcc.RadioItems(
                    id="s6-sel-method",
                    options=[
                        {"label": "Top N by importance", "value": "topn"},
                        {"label": "Cumulative importance threshold", "value": "cumul"},
                        {"label": "Manual selection", "value": "manual"},
                    ],
                    value="topn", inline=True,
                    inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "20px"},
                ),
                html.Div(id="s6-sel-controls", className="mt-2"),
                dcc.Dropdown(
                    id="s6-keep-select",
                    options=[{"label": f, "value": f} for f in importances["Feature"].tolist()],
                    value=importances.head(min(20, n_feat))["Feature"].tolist(),
                    multi=True,
                    placeholder="Features to keep...",
                ),
                html.Div(id="s6-sel-info", className="mt-2"),
            ]), className="mb-3"),

            dbc.Card(dbc.CardBody([
                html.H5("Apply RF feature selection"),
                dbc.Button("Apply RF feature selection", id="s6-apply-btn", color="primary", n_clicks=0),
                html.Div(id="s6-apply-status", className="mt-2"),
            ]), className="mb-3"),
        ])

    return html.Div([
        html.Div([
            html.H2("Stage 6 — RF Importance"),
            html.P("Rank features by Random Forest importance and select the most informative ones."),
        ], className="stage-header"),

        html.P(f"{len(numeric_cols)} features available after correlation filter.", style={"color": "#6b7280"}),

        dbc.Card(dbc.CardBody([
            html.H5("Random Forest settings"),
            dbc.Row([
                dbc.Col([html.Label("Number of trees"), dbc.Input(id="s6-n-trees", type="number", value=200, min=50, max=500, step=50)], width=3),
                dbc.Col([html.Label("Max depth (0 = unlimited)"), dbc.Input(id="s6-max-depth", type="number", value=0, min=0, max=20)], width=3),
                dbc.Col([html.Label("Random seed"), dbc.Input(id="s6-seed", type="number", value=42, min=0)], width=3),
            ]),
            html.Br(),
            dbc.Button("Train Random Forest", id="s6-train-btn", color="primary", n_clicks=0),
            dcc.Loading(html.Div(id="s6-train-status"), type="circle"),
        ]), className="mb-3"),

        imp_section,
    ])


@callback(
    Output("s6-train-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s6-train-btn", "n_clicks"),
    State("s6-n-trees", "value"),
    State("s6-max-depth", "value"),
    State("s6-seed", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def train_rf(n_clicks, n_trees, max_depth, seed, state):
    if not n_clicks:
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    y_train = server_store.get_df("y_train")
    if X_train is None or y_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    X_tr = X_train[numeric_cols]

    max_d = None if (max_depth or 0) == 0 else int(max_depth)
    rf = RandomForestClassifier(
        n_estimators=int(n_trees or 200),
        max_depth=max_d,
        random_state=int(seed or 42),
        n_jobs=-1,
    )
    rf.fit(X_tr, y_train)

    importances = pd.DataFrame({
        "Feature": X_tr.columns,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    importances["Rank"] = importances.index + 1
    importances["Cumulative %"] = (
        importances["Importance"].cumsum() / importances["Importance"].sum() * 100
    ).round(1)

    server_store.set_val("rf_importances", importances)
    server_store.set_val("rf_model", rf)
    state = dict(state or {})

    return dbc.Alert("RF trained. Reload the page / click another stage then return to see the chart.", color="success"), state


@callback(
    Output("s6-imp-chart", "figure"),
    Input("s6-topn", "value"),
    prevent_initial_call=True,
)
def update_imp_chart(top_n):
    importances = server_store.get_val("rf_importances")
    if importances is None:
        return {}
    top_n = top_n or 30
    top_df = importances.head(top_n)
    fig = px.bar(
        top_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Blues",
        text=top_df["Importance"].map(lambda v: f"{v:.4f}"),
        title=f"Top {top_n} features by RF importance",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 22 + 80),
        showlegend=False, margin=dict(t=50, b=20, l=20, r=20),
        coloraxis_showscale=False,
    )
    return fig


@callback(
    Output("s6-sel-controls", "children"),
    Output("s6-keep-select", "value"),
    Input("s6-sel-method", "value"),
    State("s6-topn", "value"),
    prevent_initial_call=True,
)
def update_sel_controls(method, top_n):
    importances = server_store.get_val("rf_importances")
    if importances is None:
        return html.Div(), []

    n_feat = len(importances)
    if method == "topn":
        keep_n = min(top_n or 30, n_feat)
        suggested = importances.head(keep_n)["Feature"].tolist()
        ctrl = html.Div([
            html.Label("Keep top N"),
            dcc.Slider(id="s6-keep-n", min=2, max=n_feat, step=1, value=keep_n,
                       marks={2:"2", n_feat//2: str(n_feat//2), n_feat: str(n_feat)}),
        ])
    elif method == "cumul":
        thresh = 90
        mask = importances["Cumulative %"] <= thresh
        if not mask.any():
            mask.iloc[0] = True
        suggested = importances.loc[mask, "Feature"].tolist()
        ctrl = html.Div([
            html.Label("Keep until cumulative importance reaches (%)"),
            dcc.Slider(id="s6-keep-n", min=50, max=100, step=5, value=thresh,
                       marks={50:"50", 90:"90", 100:"100"}),
        ])
    else:
        suggested = importances["Feature"].tolist()
        ctrl = html.Div(dcc.Slider(id="s6-keep-n", min=2, max=n_feat, step=1, value=n_feat), style={"display": "none"})

    return ctrl, suggested


@callback(
    Output("s6-sel-info", "children"),
    Input("s6-keep-select", "value"),
    prevent_initial_call=True,
)
def sel_info(features):
    importances = server_store.get_val("rf_importances")
    if importances is None or not features:
        return ""
    return dbc.Alert(f"Keeping {len(features)} of {len(importances)} features.", color="info", className="py-2")


@callback(
    Output("s6-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s6-apply-btn", "n_clicks"),
    State("s6-keep-select", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_rf_selection(n_clicks, keep_features, state):
    if not n_clicks:
        return no_update, no_update
    if not keep_features:
        return dbc.Alert("Select at least one feature.", color="danger"), no_update

    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    avail = [f for f in keep_features if f in X_train.columns]
    server_store.set_df("X_train", X_train[avail])
    server_store.set_df("X_test", X_test[avail])

    state = dict(state or {})
    state["rf_selected_features"] = avail
    state["rf_done"] = True

    return dbc.Alert(
        f"RF selection applied — {len(avail)} features kept. Proceed to Stage 7.",
        color="success",
    ), state
