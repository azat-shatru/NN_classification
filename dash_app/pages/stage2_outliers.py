"""
Stage 2 — Outlier Detection & Treatment
Per-column IQR / Z-score detection with box plots, plus Isolation Forest.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from scipy import stats as scipy_stats

from dash import dcc, html, Input, Output, State, callback, no_update, ALL, ctx
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _iqr_outliers(series: pd.Series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def _zscore_outliers(series: pd.Series, threshold=3.0):
    z = np.abs(scipy_stats.zscore(series.dropna()))
    idx = series.dropna().index[z > threshold]
    mask = pd.Series(False, index=series.index)
    mask[idx] = True
    return mask


def _winsorize(series: pd.Series):
    lo = np.percentile(series.dropna(), 1)
    hi = np.percentile(series.dropna(), 99)
    return series.clip(lower=lo, upper=hi)


def _log_transform(series: pd.Series):
    shift = max(0, -series.min()) + 1
    return np.log1p(series + shift)


def layout(state: dict) -> html.Div:
    if not state.get("df_imputed"):
        return dbc.Alert("Complete Stage 1 — Missing Values first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No train data found. Complete Stage 1 first.", color="warning")

    num_cols = [c for c in (state.get("numeric_cols") or []) if c in X_train.columns]
    if not num_cols:
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

    if not num_cols:
        return html.Div([
            html.Div([html.H2("Stage 2 — Outlier Treatment"), html.P("Detect and treat outliers in numeric columns.")], className="stage-header"),
            dbc.Alert("No numeric columns found. Proceed to Stage 2.5.", color="info"),
        ])

    # Build accordion with one item per numeric column
    accordion_items = []
    for col in num_cols:
        col_fig = go.Figure()
        col_fig.add_trace(go.Box(
            y=X_train[col].dropna(), name=col,
            boxpoints="outliers", marker_color="#2E75B6", line_color="#1F3864",
        ))
        col_fig.update_layout(height=250, margin=dict(t=20, b=20), yaxis_title=col)

        mask, _, _ = _iqr_outliers(X_train[col].dropna())
        n_out = int(mask.sum())
        pct = n_out / len(X_train) * 100
        skew = round(float(X_train[col].skew()), 3)

        accordion_items.append(
            dbc.AccordionItem(
                [
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=col_fig), width=8),
                        dbc.Col([
                            html.Div([html.Div("IQR Outliers", className="metric-label"), html.Div(n_out, className="metric-value")], className="metric-card mb-2"),
                            html.Div([html.Div("Outlier %", className="metric-label"), html.Div(f"{pct:.1f}%", className="metric-value")], className="metric-card mb-2"),
                            html.Div([html.Div("Skewness", className="metric-label"), html.Div(skew, className="metric-value")], className="metric-card mb-2"),
                            html.Label("Treatment"),
                            dcc.Dropdown(
                                id={"type": "s2-treatment", "col": col},
                                options=[
                                    {"label": "Keep", "value": "Keep"},
                                    {"label": "Winsorize (1–99 pct)", "value": "Winsorize"},
                                    {"label": "Remove rows", "value": "Remove"},
                                    {"label": "Log-transform", "value": "Log"},
                                ],
                                value="Keep",
                                clearable=False,
                            ),
                        ], width=4),
                    ])
                ],
                title=f"{col}  —  {n_out} outliers ({pct:.1f}%)",
                item_id=col,
            )
        )

    return html.Div([
        html.Div([
            html.H2("Stage 2 — Outlier Treatment"),
            html.P("Detect and treat outliers in numeric columns."),
        ], className="stage-header"),

        dbc.Card(dbc.CardBody([
            html.H5("Detection method"),
            dcc.RadioItems(
                id="s2-detection",
                options=[
                    {"label": "IQR (1.5×)", "value": "IQR"},
                    {"label": "Z-score (threshold=3)", "value": "Zscore"},
                ],
                value="IQR",
                inline=True,
                inputStyle={"marginRight": "6px"},
                labelStyle={"marginRight": "20px"},
            ),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Per-column outlier inspection"),
            dbc.Accordion(accordion_items, start_collapsed=True, always_open=False),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Multivariate — Isolation Forest"),
            html.P("Detects rows that are anomalous across ALL numeric columns combined.",
                   style={"color": "#6b7280", "fontSize": "0.85rem"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Expected outlier fraction (%)"),
                    dcc.Slider(id="s2-contamination", min=1, max=20, step=1, value=5,
                               marks={1:"1%", 5:"5%", 10:"10%", 20:"20%"}),
                ], width=8),
            ]),
            dbc.Button("Run Isolation Forest", id="s2-iso-btn", color="secondary", n_clicks=0, className="mt-2"),
            html.Div(id="s2-iso-status", className="mt-2"),
            dbc.Checklist(id="s2-iso-remove", options=[], value=[], className="mt-2"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Apply treatments"),
            dbc.Button("Apply all treatments", id="s2-apply-btn", color="primary", n_clicks=0),
            html.Div(id="s2-apply-status", className="mt-2"),
        ]), className="mb-3"),
    ])


@callback(
    Output("s2-iso-status", "children"),
    Output("s2-iso-remove", "options"),
    Input("s2-iso-btn", "n_clicks"),
    State("s2-contamination", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def run_isolation_forest(n_clicks, contamination, state):
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), []
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    if not num_cols:
        return dbc.Alert("No numeric columns.", color="warning"), []

    cont = (contamination or 5) / 100
    iso = IsolationForest(contamination=cont, random_state=42)
    preds = iso.fit_predict(X_train[num_cols].fillna(X_train[num_cols].median()))
    mask = preds == -1
    n_iso = int(mask.sum())
    server_store.set_val("iso_mask", mask.tolist())

    opts = [{"label": f"Remove these {n_iso} rows ({n_iso/len(X_train)*100:.1f}%)", "value": "remove"}]
    alert = dbc.Alert(
        f"Isolation Forest flagged {n_iso} rows ({n_iso/len(X_train)*100:.1f}%) as outliers.",
        color="warning",
    )
    return alert, opts


@callback(
    Output("s2-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s2-apply-btn", "n_clicks"),
    State({"type": "s2-treatment", "col": ALL}, "value"),
    State({"type": "s2-treatment", "col": ALL}, "id"),
    State("s2-detection", "value"),
    State("s2-iso-remove", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_treatments(n_clicks, treatment_values, treatment_ids, detection, iso_remove, state):
    if not n_clicks:
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    X_tr = X_train.copy()
    X_te = X_test.copy()
    treatments = {d["col"]: v for d, v in zip(treatment_ids, treatment_values)}
    removed_masks = []

    for col, choice in treatments.items():
        if col not in X_tr.columns:
            continue
        if choice == "Winsorize":
            X_tr[col] = _winsorize(X_tr[col])
            X_te[col] = _winsorize(X_te[col])
        elif choice == "Log":
            X_tr[col] = _log_transform(X_tr[col])
            X_te[col] = _log_transform(X_te[col])
        elif choice == "Remove":
            if detection == "IQR":
                mask, _, _ = _iqr_outliers(X_tr[col].fillna(X_tr[col].median()))
            else:
                mask = _zscore_outliers(X_tr[col].fillna(X_tr[col].median()))
            removed_masks.append(mask)

    # Apply Isolation Forest removal
    if iso_remove and "remove" in iso_remove:
        iso_mask_list = server_store.get_val("iso_mask")
        if iso_mask_list:
            iso_mask = pd.Series(iso_mask_list, index=X_tr.index[:len(iso_mask_list)])
            removed_masks.append(iso_mask.reindex(X_tr.index, fill_value=False))

    if removed_masks:
        combined = pd.concat(removed_masks, axis=1).any(axis=1)
        n_removed = int(combined.sum())
        X_tr = X_tr[~combined].reset_index(drop=True)
        y_train = server_store.get_df("y_train")
        if y_train is not None:
            y_train = y_train.reset_index(drop=True)[~combined.values].reset_index(drop=True)
            server_store.set_df("y_train", y_train)
        info = f" Removed {n_removed} outlier rows from train set."
    else:
        info = ""

    server_store.set_df("X_train", X_tr)
    server_store.set_df("X_test", X_te)

    state = dict(state or {})
    state["df_outliers_done"] = True

    cols_changed = [c for c, t in treatments.items() if t != "Keep"]
    if cols_changed:
        post_fig = px.box(X_tr[cols_changed], title="Post-treatment distributions")
        post_fig.update_layout(height=300, margin=dict(t=40, b=20))
        chart = dcc.Graph(figure=post_fig)
    else:
        chart = html.Div()

    alert = dbc.Alert(
        f"Outlier treatment applied.{info} Proceed to Stage 2.5.",
        color="success",
    )
    return html.Div([alert, chart]), state
