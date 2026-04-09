"""
Stage 8 — Combination Testing
Forward stepwise, backward stepwise, manual explorer.
Uses dcc.Interval for progress display during stepwise runs.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss
import threading

from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _score_features(X_tr, y_tr, features, cv=5):
    if not features:
        return {}
    Xf = X_tr[features]
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=500, random_state=42, multi_class="multinomial", solver="lbfgs")
    acc_scores = cross_val_score(lr, Xf, y_tr, cv=cv_obj, scoring="accuracy")
    auc_scores = cross_val_score(lr, Xf, y_tr, cv=cv_obj, scoring="roc_auc_ovr_weighted")

    lr_full = LogisticRegression(max_iter=500, random_state=42, multi_class="multinomial", solver="lbfgs")
    lr_full.fit(Xf, y_tr)
    ll_full = -log_loss(y_tr, lr_full.predict_proba(Xf), normalize=False)
    classes, counts = np.unique(y_tr, return_counts=True)
    null_proba = np.tile(counts / counts.sum(), (len(y_tr), 1))
    ll_null = -log_loss(y_tr, null_proba, normalize=False)
    mcfadden = round(1 - (ll_full / ll_null), 4) if ll_null != 0 else float("nan")

    return {
        "acc_mean": round(float(np.mean(acc_scores)), 4),
        "acc_std":  round(float(np.std(acc_scores)), 4),
        "auc_mean": round(float(np.mean(auc_scores)), 4),
        "mcfadden": mcfadden,
        "n_features": len(features),
        "features": features,
    }


def layout(state: dict) -> html.Div:
    if not state.get("factor_done"):
        return dbc.Alert("Complete Stage 7 — Factor Analysis first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    all_features = list(X_train.columns)
    comb_results = server_store.get_val("comb_results") or []
    selected_features = state.get("selected_features") or []

    # Results table
    if comb_results:
        res_df = pd.DataFrame([{
            "Label": r.get("label", ""),
            "N features": r.get("n_features", 0),
            "CV Accuracy": r.get("acc_mean", ""),
            "CV AUC": r.get("auc_mean", ""),
            "McFadden R²": r.get("mcfadden", ""),
        } for r in comb_results])
        results_tbl = dbc.Table.from_dataframe(
            res_df.sort_values("CV Accuracy", ascending=False) if "CV Accuracy" in res_df else res_df,
            striped=True, hover=True, size="sm",
        )
    else:
        results_tbl = html.P("No combinations scored yet.", style={"color": "#6b7280"})

    return html.Div([
        html.Div([
            html.H2("Stage 8 — Combination Testing"),
            html.P("Compare feature combinations — find the minimal set that maximises performance."),
        ], className="stage-header"),

        html.P(f"{len(all_features)} features available.", style={"color": "#6b7280"}),

        dbc.Tabs([
            dbc.Tab(label="Forward Stepwise", tab_id="fw", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Forward Stepwise Selection"),
                    html.P("Starts with no features; adds the feature that most improves CV accuracy each step.",
                           style={"color": "#6b7280", "fontSize": "0.85rem"}),
                    dbc.Row([
                        dbc.Col([html.Label("Optimise by"), dcc.Dropdown(
                            id="s8-fw-metric",
                            options=[{"label": m, "value": m} for m in ["acc_mean", "auc_mean", "mcfadden"]],
                            value="acc_mean", clearable=False,
                        )], width=3),
                        dbc.Col([html.Label("CV folds"), dcc.Slider(id="s8-fw-cv", min=3, max=10, step=1, value=5, marks={3:"3",5:"5",10:"10"})], width=4),
                        dbc.Col([html.Label("Max features to add"), dcc.Slider(
                            id="s8-fw-max", min=1, max=min(len(all_features), 20), step=1,
                            value=min(15, len(all_features)),
                            marks={1:"1", min(15, len(all_features)):str(min(15, len(all_features)))},
                        )], width=4),
                    ]),
                    html.Br(),
                    dbc.Button("Run forward stepwise", id="s8-fw-btn", color="primary", n_clicks=0),
                    dcc.Loading(html.Div(id="s8-fw-status"), type="circle"),
                    dcc.Graph(id="s8-fw-chart"),
                    html.Div(id="s8-fw-results"),
                ]), className="mt-2"),
            ]),

            dbc.Tab(label="Backward Stepwise", tab_id="bw", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Backward Stepwise Elimination"),
                    html.P("Starts with all features; removes the least useful one each step.",
                           style={"color": "#6b7280", "fontSize": "0.85rem"}),
                    dbc.Row([
                        dbc.Col([html.Label("Optimise by"), dcc.Dropdown(
                            id="s8-bw-metric",
                            options=[{"label": m, "value": m} for m in ["acc_mean", "auc_mean", "mcfadden"]],
                            value="acc_mean", clearable=False,
                        )], width=3),
                        dbc.Col([html.Label("CV folds"), dcc.Slider(id="s8-bw-cv", min=3, max=10, step=1, value=5, marks={3:"3",5:"5",10:"10"})], width=4),
                        dbc.Col([html.Label("Min features to retain"), dcc.Slider(
                            id="s8-bw-min", min=1, max=max(2, len(all_features)-1), step=1,
                            value=max(2, len(all_features)-10),
                            marks={1:"1", max(2,len(all_features)-10):str(max(2,len(all_features)-10))},
                        )], width=4),
                    ]),
                    html.Br(),
                    dbc.Button("Run backward stepwise", id="s8-bw-btn", color="primary", n_clicks=0),
                    dcc.Loading(html.Div(id="s8-bw-status"), type="circle"),
                    dcc.Graph(id="s8-bw-chart"),
                    html.Div(id="s8-bw-results"),
                ]), className="mt-2"),
            ]),

            dbc.Tab(label="Manual Explorer", tab_id="manual", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Manual Feature Explorer"),
                    html.P("Pick any combination and score it instantly.",
                           style={"color": "#6b7280", "fontSize": "0.85rem"}),
                    dcc.Dropdown(
                        id="s8-manual-sel",
                        options=[{"label": f, "value": f} for f in all_features],
                        value=[f for f in (selected_features or all_features[:10]) if f in all_features],
                        multi=True,
                    ),
                    dbc.Row([
                        dbc.Col([html.Label("CV folds"), dcc.Slider(id="s8-man-cv", min=3, max=10, step=1, value=5, marks={3:"3",5:"5",10:"10"})], width=4),
                    ], className="mt-2"),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Button("Score this combination", id="s8-score-btn", color="secondary", n_clicks=0), width="auto"),
                        dbc.Col(dbc.Button("Use this combination for Stage 9", id="s8-use-manual-btn", color="success", n_clicks=0), width="auto"),
                    ], className="g-2"),
                    dcc.Loading(html.Div(id="s8-manual-status"), type="circle"),
                ]), className="mt-2"),
            ]),
        ], id="s8-tabs", active_tab="fw"),

        html.Hr(className="dash-divider"),

        dbc.Card(dbc.CardBody([
            html.H5("All scored combinations"),
            results_tbl,
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Finalise feature set for Stage 9"),
            html.Div(
                dbc.Alert(
                    f"{len(selected_features)} features selected: {', '.join(selected_features[:10])}{'…' if len(selected_features)>10 else ''}",
                    color="success",
                ) if selected_features else
                dbc.Alert("No feature set finalised yet.", color="info")
            ),
        ]), className="mb-3"),
    ])


@callback(
    Output("s8-fw-status", "children"),
    Output("s8-fw-chart", "figure"),
    Output("s8-fw-results", "children"),
    Input("s8-fw-btn", "n_clicks"),
    State("s8-fw-metric", "value"),
    State("s8-fw-cv", "value"),
    State("s8-fw-max", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def run_forward(n_clicks, metric, cv, max_feat, state):
    if not n_clicks:
        return no_update, no_update, no_update
    X_train = server_store.get_df("X_train")
    y_train = server_store.get_df("y_train")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), {}, html.Div()

    all_features = list(X_train.columns)
    selected, remaining = [], all_features.copy()
    history = []
    metric = metric or "acc_mean"

    for step in range(int(max_feat or 15)):
        best_score = -np.inf
        best_feat = None
        best_s = None
        for feat in remaining:
            candidate = selected + [feat]
            s = _score_features(X_train, y_train, candidate, cv=int(cv or 5))
            if s and s.get(metric, -np.inf) > best_score:
                best_score = s[metric]
                best_feat = feat
                best_s = s
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append({**best_s, "step": step+1, "added": best_feat})

    server_store.set_val("fw_history", history)

    if not history:
        return dbc.Alert("No steps completed.", color="warning"), {}, html.Div()

    hist_df = pd.DataFrame(history)
    fig = px.line(hist_df, x="step", y=metric, markers=True,
                  title="Forward stepwise — score by step",
                  labels={"step": "Step (features added)", metric: metric})
    fig.update_layout(height=300, margin=dict(t=40, b=20))

    best_row = hist_df.loc[hist_df[metric].idxmax()]
    results = html.Div([
        dbc.Table.from_dataframe(
            hist_df[["step","added","acc_mean","auc_mean","mcfadden","n_features"]],
            striped=True, hover=True, size="sm",
        ),
        dbc.Button(
            f"Use forward result ({int(best_row['n_features'])} features)",
            id="s8-use-fw-btn", color="success", n_clicks=0, className="mt-2",
        ),
    ])
    return dbc.Alert(f"Forward stepwise complete — {len(selected)} features.", color="success"), fig, results


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("s8-use-fw-btn", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def use_fw(n_clicks, state):
    if not n_clicks:
        return no_update
    history = server_store.get_val("fw_history")
    if not history:
        return no_update
    hist_df = pd.DataFrame(history)
    best_row = hist_df.loc[hist_df["acc_mean"].idxmax()]
    state = dict(state or {})
    state["selected_features"] = best_row["features"]
    return state


@callback(
    Output("s8-bw-status", "children"),
    Output("s8-bw-chart", "figure"),
    Output("s8-bw-results", "children"),
    Input("s8-bw-btn", "n_clicks"),
    State("s8-bw-metric", "value"),
    State("s8-bw-cv", "value"),
    State("s8-bw-min", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def run_backward(n_clicks, metric, cv, min_feat, state):
    if not n_clicks:
        return no_update, no_update, no_update
    X_train = server_store.get_df("X_train")
    y_train = server_store.get_df("y_train")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), {}, html.Div()

    all_features = list(X_train.columns)
    remaining = all_features.copy()
    history = []
    metric = metric or "acc_mean"
    step = 0

    while len(remaining) > int(min_feat or 2):
        best_score = -np.inf
        worst_feat = None
        best_s = None
        for feat in remaining:
            candidate = [f for f in remaining if f != feat]
            s = _score_features(X_train, y_train, candidate, cv=int(cv or 5))
            if s and s.get(metric, -np.inf) > best_score:
                best_score = s[metric]
                worst_feat = feat
                best_s = s
        if worst_feat is None:
            break
        remaining.remove(worst_feat)
        step += 1
        history.append({**best_s, "step": step, "removed": worst_feat})

    server_store.set_val("bw_history", history)

    if not history:
        return dbc.Alert("No steps completed.", color="warning"), {}, html.Div()

    hist_df = pd.DataFrame(history)
    fig = px.line(hist_df, x="n_features", y=metric, markers=True,
                  title="Backward stepwise — score vs. feature count",
                  labels={"n_features": "Features remaining", metric: metric})
    fig.update_layout(height=300, margin=dict(t=40, b=20))

    best_row = hist_df.loc[hist_df[metric].idxmax()]
    results = html.Div([
        dbc.Table.from_dataframe(
            hist_df[["step","removed","acc_mean","auc_mean","mcfadden","n_features"]],
            striped=True, hover=True, size="sm",
        ),
        dbc.Button(
            f"Use backward result ({int(best_row['n_features'])} features)",
            id="s8-use-bw-btn", color="success", n_clicks=0, className="mt-2",
        ),
    ])
    return dbc.Alert(f"Backward stepwise complete — {len(remaining)} features remain.", color="success"), fig, results


@callback(
    Output("app-state", "data", allow_duplicate=True),
    Input("s8-use-bw-btn", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def use_bw(n_clicks, state):
    if not n_clicks:
        return no_update
    history = server_store.get_val("bw_history")
    if not history:
        return no_update
    hist_df = pd.DataFrame(history)
    best_row = hist_df.loc[hist_df["acc_mean"].idxmax()]
    state = dict(state or {})
    state["selected_features"] = best_row["features"]
    return state


@callback(
    Output("s8-manual-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s8-score-btn", "n_clicks"),
    Input("s8-use-manual-btn", "n_clicks"),
    State("s8-manual-sel", "value"),
    State("s8-man-cv", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def manual_actions(n_score, n_use, features, cv, state):
    triggered = ctx.triggered_id
    state = dict(state or {})

    if triggered == "s8-use-manual-btn":
        if not features:
            return dbc.Alert("Select features first.", color="warning"), no_update
        state["selected_features"] = features
        comb_results = server_store.get_val("comb_results") or []
        return dbc.Alert(f"{len(features)} features selected for Stage 9.", color="success"), state

    if triggered == "s8-score-btn":
        if not features:
            return dbc.Alert("Select at least one feature.", color="warning"), no_update
        X_train = server_store.get_df("X_train")
        y_train = server_store.get_df("y_train")
        if X_train is None:
            return dbc.Alert("No data.", color="danger"), no_update
        s = _score_features(X_train, y_train, features, cv=int(cv or 5))
        comb_results = server_store.get_val("comb_results") or []
        comb_results.append({
            **s,
            "label": f"Manual ({len(features)} feat)",
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        })
        server_store.set_val("comb_results", comb_results)
        return dbc.Alert(
            f"Accuracy: {s['acc_mean']:.4f} ± {s['acc_std']:.4f}  |  "
            f"AUC: {s['auc_mean']:.4f}  |  McFadden R²: {s['mcfadden']:.4f}",
            color="success",
        ), no_update

    return no_update, no_update
