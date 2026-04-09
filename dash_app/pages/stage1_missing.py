"""
Stage 1 — Missing Value Treatment
Train/test split happens here. Imputation fitted on train only.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _split_df(df, target, test_size=0.2, seed=42):
    X = df.drop(columns=[target])
    y = df[target]
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
    return X_tr, X_te, y_tr, y_te


def _apply_imputer(X_train, X_test, cols, strategy, n_neighbors=5):
    if strategy == "knn":
        imp = KNNImputer(n_neighbors=n_neighbors)
    else:
        imp = SimpleImputer(strategy=strategy)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols] = imp.fit_transform(X_train[cols])
    X_test[cols] = imp.transform(X_test[cols])
    return X_train, X_test


def layout(state: dict) -> html.Div:
    df = server_store.get_df("df")
    if df is None:
        return dbc.Alert("Complete Stage 0 — upload a dataset first.", color="warning")

    target = state.get("target_col")
    if not target or target not in df.columns:
        return dbc.Alert("Set a target column in Stage 0 first.", color="warning")

    X_train = server_store.get_df("X_train")
    split_done = X_train is not None

    missing_table = html.Div()
    if split_done:
        miss = X_train.isnull().sum()
        miss_pct = (miss / len(X_train) * 100).round(1)
        miss_df = pd.DataFrame({
            "Column": miss.index,
            "Missing count": miss.values,
            "Missing %": miss_pct.values,
        }).query("`Missing count` > 0").sort_values("Missing %", ascending=False)

        if miss_df.empty:
            missing_table = dbc.Alert("No missing values in the train set. Proceed to Stage 2.", color="success")
        else:
            rows = miss_df.reset_index(drop=True).to_dict("records")
            missing_table = html.Div([
                dbc.Alert(f"{len(miss_df)} columns have missing values in the train set.", color="warning"),
                html.Div(
                    style={"maxHeight": "300px", "overflowY": "auto", "border": "1px solid #e5e7eb", "borderRadius": "6px"},
                    children=dbc.Table.from_dataframe(miss_df.reset_index(drop=True), striped=True, hover=True, size="sm"),
                ),
            ])

    return html.Div([
        html.Div([
            html.H2("Stage 1 — Missing Values"),
            html.P("Train/test split happens here — imputation is fitted on train only."),
        ], className="stage-header"),

        # Step 1: split
        dbc.Card(dbc.CardBody([
            html.H5("Step 1 — Train / Test Split"),
            dbc.Alert("Split happens before imputation to prevent data leakage.", color="info", className="py-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Test set size (%)"),
                    dcc.Slider(id="s1-test-size", min=10, max=40, step=5, value=20,
                               marks={10:"10%", 20:"20%", 30:"30%", 40:"40%"}),
                ], width=6),
                dbc.Col([
                    html.Label("Random seed"),
                    dbc.Input(id="s1-seed", type="number", value=42, min=0),
                ], width=3),
            ]),
            html.Br(),
            dbc.Button("Perform split", id="s1-split-btn", color="primary", n_clicks=0),
            html.Div(id="s1-split-status", className="mt-2"),
        ]), className="mb-3"),

        # Split summary
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Train rows", className="metric-label"), html.Div(id="s1-train-rows", className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Test rows", className="metric-label"), html.Div(id="s1-test-rows", className="metric-value")]), className="metric-card"), width=3),
        ], className="mb-3"),

        # Step 2: missing summary
        dbc.Card(dbc.CardBody([
            html.H5("Step 2 — Missing value summary (train set)"),
            html.Div(id="s1-missing-table", children=missing_table),
        ]), className="mb-3"),

        # Steps 3-5: always in DOM, hidden until split is done
        html.Div([
            # Step 3: drop high-missing
            dbc.Card(dbc.CardBody([
                html.H5("Step 3 — Drop columns with too many missing values"),
                html.Label("Drop threshold (%)"),
                dcc.Slider(id="s1-drop-thresh", min=10, max=90, step=5, value=40,
                           marks={10:"10%", 40:"40%", 90:"90%"}),
                html.Div(id="s1-drop-candidates"),
                dcc.Dropdown(id="s1-drop-confirm", multi=True, placeholder="Columns to drop (deselect to keep)"),
            ]), className="mb-3"),

            # Step 4: imputation strategy
            dbc.Card(dbc.CardBody([
                html.H5("Step 4 — Imputation strategy"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Numeric columns strategy"),
                        dcc.Dropdown(
                            id="s1-num-strategy",
                            options=[
                                {"label": "Median (robust to outliers)", "value": "median"},
                                {"label": "Mean", "value": "mean"},
                                {"label": "KNN", "value": "knn"},
                            ],
                            value="median", clearable=False,
                        ),
                        html.Div(id="s1-knn-div"),
                    ], width=4),
                    dbc.Col([
                        html.Label("Categorical/Ordinal columns strategy"),
                        dcc.Dropdown(
                            id="s1-cat-strategy",
                            options=[
                                {"label": "Most frequent (mode)", "value": "most_frequent"},
                                {"label": "Constant (Unknown)", "value": "constant"},
                            ],
                            value="most_frequent", clearable=False,
                        ),
                    ], width=4),
                ]),
            ]), className="mb-3"),

            # Step 5: apply
            dbc.Card(dbc.CardBody([
                html.H5("Step 5 — Apply imputation"),
                dbc.Button("Apply imputation", id="s1-apply-btn", color="primary", n_clicks=0),
                html.Div(id="s1-apply-status", className="mt-2"),
            ]), className="mb-3"),
        ], style={"display": "block" if split_done else "none"}, id="s1-post-split-section"),
    ])


@callback(
    Output("s1-split-status", "children"),
    Output("s1-train-rows", "children"),
    Output("s1-test-rows", "children"),
    Output("s1-missing-table", "children"),
    Output("s1-drop-candidates", "children"),
    Output("s1-drop-confirm", "options"),
    Output("s1-drop-confirm", "value"),
    Output("s1-post-split-section", "style"),
    Input("s1-split-btn", "n_clicks"),
    State("s1-test-size", "value"),
    State("s1-seed", "value"),
    State("s1-drop-thresh", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def perform_split(n_clicks, test_size_pct, seed, drop_thresh, state):
    df = server_store.get_df("df")
    if df is None:
        return dbc.Alert("No data loaded.", color="danger"), "", "", no_update, no_update, [], [], no_update
    target = (state or {}).get("target_col")
    if not target or target not in df.columns:
        return dbc.Alert("Target column missing.", color="danger"), "", "", no_update, no_update, [], [], no_update

    X_tr, X_te, y_tr, y_te = _split_df(df, target, (test_size_pct or 20) / 100, int(seed or 42))
    server_store.set_df("X_train", X_tr)
    server_store.set_df("X_test", X_te)
    server_store.set_df("y_train", y_tr)
    server_store.set_df("y_test", y_te)

    miss = X_tr.isnull().sum()
    miss_pct = (miss / len(X_tr) * 100).round(1)
    miss_df = pd.DataFrame({
        "Column": miss.index, "Missing count": miss.values, "Missing %": miss_pct.values,
    }).query("`Missing count` > 0").sort_values("Missing %", ascending=False)

    if miss_df.empty:
        miss_tbl = dbc.Alert("No missing values in the train set.", color="success")
        drop_info = dbc.Alert("No high-missing columns.", color="success")
        drop_opts, drop_vals = [], []
    else:
        miss_tbl = html.Div([
            dbc.Alert(f"{len(miss_df)} columns have missing values.", color="warning"),
            dbc.Table.from_dataframe(miss_df.reset_index(drop=True), striped=True, hover=True, size="sm"),
        ])
        thresh = drop_thresh or 40
        candidates = miss_df.loc[miss_df["Missing %"] > thresh, "Column"].tolist()
        drop_opts = [{"label": c, "value": c} for c in candidates]
        drop_vals = candidates
        if candidates:
            drop_info = dbc.Alert(f"Columns exceeding {thresh}%: {', '.join(candidates)}", color="warning")
        else:
            drop_info = dbc.Alert(f"No columns exceed the {thresh}% threshold.", color="success")

    status = dbc.Alert(f"Split complete — Train: {len(X_tr)} rows | Test: {len(X_te)} rows", color="success")
    return status, len(X_tr), len(X_te), miss_tbl, drop_info, drop_opts, drop_vals, {"display": "block"}


@callback(
    Output("s1-knn-div", "children"),
    Input("s1-num-strategy", "value"),
    prevent_initial_call=True,
)
def show_knn_slider(strategy):
    if strategy == "knn":
        return html.Div([
            html.Label("KNN neighbours (k)", className="mt-2"),
            dcc.Slider(id="s1-knn-k", min=3, max=15, step=1, value=5,
                       marks={3:"3", 5:"5", 10:"10", 15:"15"}),
        ])
    return html.Div(dcc.Slider(id="s1-knn-k", min=3, max=15, step=1, value=5), style={"display":"none"})


@callback(
    Output("s1-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s1-apply-btn", "n_clicks"),
    State("s1-num-strategy", "value"),
    State("s1-cat-strategy", "value"),
    State("s1-knn-k", "value"),
    State("s1-drop-confirm", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_imputation(n_clicks, num_strategy, cat_strategy, knn_k, drop_cols, state):
    if not n_clicks:
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("Perform the split first.", color="warning"), no_update

    state = dict(state or {})
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Drop high-missing columns if confirmed
    if drop_cols:
        X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
        X_te = X_te.drop(columns=drop_cols, errors="ignore")
        for key in ["numeric_cols", "categorical_cols", "ordinal_cols"]:
            state[key] = [c for c in (state.get(key) or []) if c not in drop_cols]

    num_cols = [c for c in (state.get("numeric_cols") or [])
                if c in X_tr.columns and X_tr[c].isnull().any()]
    cat_cols = [c for c in (state.get("categorical_cols") or []) + (state.get("ordinal_cols") or [])
                if c in X_tr.columns and X_tr[c].isnull().any()]

    if num_cols and num_strategy:
        k = int(knn_k or 5)
        X_tr, X_te = _apply_imputer(X_tr, X_te, num_cols, num_strategy, n_neighbors=k)

    if cat_cols and cat_strategy:
        fill_val = "Unknown" if cat_strategy == "constant" else None
        imp = SimpleImputer(strategy=cat_strategy, fill_value=fill_val)
        X_tr[cat_cols] = imp.fit_transform(X_tr[cat_cols])
        X_te[cat_cols] = imp.transform(X_te[cat_cols])

    server_store.set_df("X_train", X_tr)
    server_store.set_df("X_test", X_te)
    state["df_imputed"] = True
    state["split_done"] = True

    remaining = int(X_tr.isnull().sum().sum())
    if remaining == 0:
        alert = dbc.Alert("All missing values resolved. Proceed to Stage 2.", color="success")
    else:
        alert = dbc.Alert(f"{remaining} missing values remain — review strategy.", color="warning")
    return alert, state
