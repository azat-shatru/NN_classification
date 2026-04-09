"""
Stage 3 — Categorical Encoding
Encoders fitted on train only, then applied to both splits.
"""
import pandas as pd
import numpy as np

from dash import dcc, html, Input, Output, State, callback, no_update, ALL, ctx
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _one_hot(X_train, X_test, cols):
    X_tr = pd.get_dummies(X_train, columns=cols, drop_first=False, dtype=int)
    X_te = pd.get_dummies(X_test, columns=cols, drop_first=False, dtype=int)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr, X_te


def _target_encode(X_train, X_test, y_train, cols):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    global_mean = float(pd.to_numeric(y_train, errors="coerce").mean())
    maps = {}
    for col in cols:
        tmp = X_train[[col]].copy()
        tmp["__y"] = y_train.values
        enc_map = tmp.groupby(col)["__y"].mean().to_dict()
        maps[col] = enc_map
        X_tr[col] = X_train[col].map(enc_map).fillna(global_mean)
        X_te[col] = X_test[col].map(enc_map).fillna(global_mean)
    return X_tr, X_te


def _ordinal_encode(X_train, X_test, col, order):
    rank_map = {v: i for i, v in enumerate(order)}
    fallback = len(order)
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[col] = X_train[col].map(rank_map).fillna(fallback).astype(int)
    X_te[col] = X_test[col].map(rank_map).fillna(fallback).astype(int)
    return X_tr, X_te


def _label_encode(X_train, X_test, col):
    X_tr = X_train.copy()
    X_te = X_test.copy()
    cats = sorted(X_train[col].dropna().unique().tolist(), key=str)
    cat_map = {v: i for i, v in enumerate(cats)}
    X_tr[col] = X_train[col].map(cat_map)
    X_te[col] = X_test[col].map(cat_map).fillna(-1).astype(int)
    return X_tr, X_te


def layout(state: dict) -> html.Div:
    if server_store.get_df("X_train") is None:
        return dbc.Alert("Complete Stage 1 — Missing Values (including split) first.", color="warning")

    X_train = server_store.get_df("X_train")
    num_cols = list(state.get("numeric_cols") or [])
    cat_cols = list(state.get("categorical_cols") or [])
    ord_cols = list(state.get("ordinal_cols") or [])

    # Limit to columns in X_train
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    ord_cols = [c for c in ord_cols if c in X_train.columns]

    # Auto-classify unassigned columns
    unassigned = [c for c in X_train.columns if c not in num_cols + cat_cols + ord_cols]
    for c in unassigned:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    strats = dict(state.get("enc_strategies") or {})
    ord_orders = dict(state.get("ord_orders") or {})

    # Build cat column controls
    cat_rows = []
    for col in cat_cols:
        n_unique = X_train[col].nunique()
        is_binary = n_unique <= 2
        ohe_thresh = 10
        default = ("Binary (keep)" if is_binary else
                   "One-Hot" if n_unique <= ohe_thresh else "Target Encoding")
        options = (["Binary (keep)"] if is_binary else
                   ["One-Hot", "Target Encoding", "Label Encoding", "Binary (keep)"])
        curr = strats.get(col, default)
        if curr not in options:
            curr = options[0]
        cat_rows.append(dbc.Row([
            dbc.Col(html.Div([
                html.Strong(col),
                html.Span(f" ({n_unique} unique)", style={"color": "#6b7280", "fontSize": "0.82rem"}),
            ]), width=4),
            dbc.Col(
                dcc.Dropdown(
                    id={"type": "s3-cat-strat", "col": col},
                    options=[{"label": o, "value": o} for o in options],
                    value=curr, clearable=False,
                ),
                width=4,
            ),
        ], className="mb-1"))

    ord_rows = []
    for col in ord_cols:
        n_unique = X_train[col].nunique()
        is_binary = n_unique <= 2
        options = (["Binary (keep)"] if is_binary else
                   ["Ordinal Encoding", "Label Encoding", "One-Hot", "Target Encoding"])
        curr = strats.get(col, options[0])
        if curr not in options:
            curr = options[0]
        cats = sorted(X_train[col].dropna().unique().tolist(), key=str)
        curr_order = ord_orders.get(col, cats)
        curr_order = [v for v in curr_order if v in cats] + [v for v in cats if v not in curr_order]
        ord_rows.append(html.Div([
            dbc.Row([
                dbc.Col(html.Strong(col), width=4),
                dbc.Col(dcc.Dropdown(
                    id={"type": "s3-ord-strat", "col": col},
                    options=[{"label": o, "value": o} for o in options],
                    value=curr, clearable=False,
                ), width=4),
            ], className="mb-1"),
            dbc.Collapse(
                dbc.Textarea(
                    id={"type": "s3-ord-order", "col": col},
                    value="\n".join(str(v) for v in curr_order),
                    style={"height": f"{min(150, len(cats)*22+30)}px", "fontFamily": "monospace"},
                    placeholder="One category per line (lowest → highest)",
                ),
                id={"type": "s3-ord-order-collapse", "col": col},
                is_open=(curr == "Ordinal Encoding"),
            ),
        ], className="mb-2"))

    return html.Div([
        html.Div([
            html.H2("Stage 3 — Encoding"),
            html.P("Encode categorical/ordinal columns. Encoders fitted on train set only."),
        ], className="stage-header"),

        # Summary
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Numeric (passthrough)", className="metric-label"), html.Div(len(num_cols), className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Categorical", className="metric-label"), html.Div(len(cat_cols), className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Ordinal", className="metric-label"), html.Div(len(ord_cols), className="metric-value")]), className="metric-card"), width=3),
        ], className="mb-3"),

        # OHE threshold slider
        dbc.Card(dbc.CardBody([
            html.H5("OHE cardinality threshold"),
            html.P("Columns with unique values above this threshold will be suggested for Target Encoding.",
                   style={"color": "#6b7280", "fontSize": "0.85rem"}),
            dcc.Slider(id="s3-ohe-thresh", min=2, max=30, step=1, value=10,
                       marks={2:"2", 10:"10", 20:"20", 30:"30"}),
            html.Div(id="s3-ohe-thresh-info"),
        ]), className="mb-3"),

        # Categorical strategies
        dbc.Card(dbc.CardBody([
            html.H5("Categorical columns — encoding strategy"),
            html.Div(cat_rows if cat_rows else dbc.Alert("No categorical columns.", color="info")),
        ]), className="mb-3") if cat_cols else html.Div(),

        # Ordinal strategies
        dbc.Card(dbc.CardBody([
            html.H5("Ordinal columns — encoding strategy"),
            html.Div(ord_rows if ord_rows else dbc.Alert("No ordinal columns.", color="info")),
        ]), className="mb-3") if ord_cols else html.Div(),

        # Apply
        dbc.Card(dbc.CardBody([
            html.H5("Apply encoding"),
            dbc.Button("Apply encoding", id="s3-apply-btn", color="primary", n_clicks=0),
            html.Div(id="s3-apply-status", className="mt-2"),
        ]), className="mb-3"),
    ])


@callback(
    Output("s3-ohe-thresh-info", "children"),
    Input("s3-ohe-thresh", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def ohe_thresh_info(thresh, state):
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return ""
    cat_cols = list((state or {}).get("categorical_cols") or [])
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    ohe_suggested = [c for c in cat_cols if X_train[c].nunique() <= (thresh or 10)]
    tgt_suggested = [c for c in cat_cols if X_train[c].nunique() > (thresh or 10)]
    return dbc.Alert(
        f"With threshold={thresh}: {len(ohe_suggested)} columns suggested OHE, {len(tgt_suggested)} suggested Target Encoding.",
        color="info", className="py-2",
    )


@callback(
    Output("s3-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s3-apply-btn", "n_clicks"),
    State({"type": "s3-cat-strat", "col": ALL}, "value"),
    State({"type": "s3-cat-strat", "col": ALL}, "id"),
    State({"type": "s3-ord-strat", "col": ALL}, "value"),
    State({"type": "s3-ord-strat", "col": ALL}, "id"),
    State({"type": "s3-ord-order", "col": ALL}, "value"),
    State({"type": "s3-ord-order", "col": ALL}, "id"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_encoding(n_clicks, cat_values, cat_ids, ord_values, ord_ids, ord_order_values, ord_order_ids, state):
    if not n_clicks:
        return no_update, no_update

    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    y_train = server_store.get_df("y_train")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    state = dict(state or {})
    cat_strats = {d["col"]: v for d, v in zip(cat_ids, cat_values)}
    ord_strats = {d["col"]: v for d, v in zip(ord_ids, ord_values)}
    ord_orders = {d["col"]: [x.strip() for x in (v or "").splitlines() if x.strip()]
                  for d, v in zip(ord_order_ids, ord_order_values)}

    strats = {**cat_strats, **ord_strats}
    state["enc_strategies"] = strats
    state["ord_orders"] = ord_orders

    X_tr = X_train.copy()
    X_te = X_test.copy()
    enc_log = []

    cat_cols = list(state.get("categorical_cols") or [])
    ord_cols = list(state.get("ordinal_cols") or [])
    cat_cols = [c for c in cat_cols if c in X_tr.columns]
    ord_cols = [c for c in ord_cols if c in X_tr.columns]
    all_enc_cols = cat_cols + ord_cols

    ohe_cols = [c for c in all_enc_cols if strats.get(c) == "One-Hot"]
    tgt_cols = [c for c in all_enc_cols if strats.get(c) == "Target Encoding"]
    lbl_cols = [c for c in all_enc_cols if strats.get(c) == "Label Encoding"]
    ord_enc_cols = [c for c in all_enc_cols if strats.get(c) == "Ordinal Encoding"]

    if ohe_cols:
        before = X_tr.shape[1]
        X_tr, X_te = _one_hot(X_tr, X_te, ohe_cols)
        enc_log.append(f"One-Hot: {ohe_cols} → +{X_tr.shape[1]-before} dummy columns")

    if tgt_cols and y_train is not None:
        X_tr, X_te = _target_encode(X_tr, X_te, y_train, tgt_cols)
        enc_log.append(f"Target Encoding: {tgt_cols}")

    for col in lbl_cols:
        X_tr, X_te = _label_encode(X_tr, X_te, col)
        enc_log.append(f"Label Encoding: {col}")

    for col in ord_enc_cols:
        order = ord_orders.get(col) or sorted(X_train[col].dropna().unique().tolist(), key=str)
        X_tr, X_te = _ordinal_encode(X_tr, X_te, col, order)
        enc_log.append(f"Ordinal Encoding: {col} ({len(order)} levels)")

    server_store.set_df("X_train", X_tr)
    server_store.set_df("X_test", X_te)
    state["df_encoded"] = True
    state["numeric_cols"] = list(X_tr.columns)
    state["categorical_cols"] = []
    state["ordinal_cols"] = []

    non_numeric = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c])]
    result_items = [html.Li(log) for log in enc_log]

    return html.Div([
        dbc.Alert(f"Encoding applied — Train: {X_tr.shape[0]}×{X_tr.shape[1]}, Test: {X_te.shape[0]}×{X_te.shape[1]}", color="success"),
        html.Ul(result_items) if result_items else html.Div(),
        dbc.Alert(f"Non-numeric columns remaining: {non_numeric}", color="warning") if non_numeric else
        dbc.Alert("All columns are numeric. Ready for Stage 4.", color="success"),
    ]), state
