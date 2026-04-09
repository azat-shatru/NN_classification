"""
Stage 7 — Factor Analysis
Bartlett/KMO tests, scree plot, loadings table with interactive threshold slider.
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


def _check_fa():
    try:
        from factor_analyzer import FactorAnalyzer  # noqa
        return True
    except ImportError:
        return False


def layout(state: dict) -> html.Div:
    if not state.get("rf_done"):
        return dbc.Alert("Complete Stage 6 — RF Importance first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    max_factors = min(len(numeric_cols) - 1, 15)

    fa_loadings = server_store.get_val("fa_loadings")
    fa_variance = server_store.get_val("fa_variance")
    fa_ev = server_store.get_val("fa_eigenvalues")
    bartlett = server_store.get_val("fa_bartlett")
    kmo = server_store.get_val("fa_kmo")

    # Scree plot
    scree_fig = {}
    if fa_ev:
        ev = fa_ev
        scree_fig_obj = go.Figure()
        scree_fig_obj.add_trace(go.Scatter(
            x=list(range(1, len(ev)+1)), y=ev, mode="lines+markers",
            marker=dict(size=8), line=dict(color="#2E75B6"),
        ))
        scree_fig_obj.add_hline(y=1.0, line_dash="dash", line_color="red",
                                annotation_text="Kaiser criterion (eigenvalue=1)")
        scree_fig_obj.update_layout(title="Scree Plot", xaxis_title="Factor",
                                    yaxis_title="Eigenvalue", height=350, margin=dict(t=50, b=30))
        scree_fig = scree_fig_obj

    return html.Div([
        html.Div([
            html.H2("Stage 7 — Factor Analysis"),
            html.P("Exploratory Factor Analysis — group correlated features into latent factors."),
        ], className="stage-header"),

        html.P(f"{len(numeric_cols)} features entering factor analysis.", style={"color": "#6b7280"}),

        # Skip option
        dbc.Card(dbc.CardBody([
            dbc.Checklist(
                id="s7-skip",
                options=[{"label": "Skip factor analysis — proceed with current features", "value": "skip"}],
                value=["skip"] if state.get("factor_done") and not fa_loadings else [],
            ),
            html.Div(id="s7-skip-status"),
        ]), className="mb-3"),

        # Step 1: suitability tests
        dbc.Card(dbc.CardBody([
            html.H5("Step 1 — Suitability tests"),
            dbc.Button("Run Bartlett & KMO tests", id="s7-tests-btn", color="secondary", n_clicks=0),
            html.Div(id="s7-tests-status"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Bartlett χ²", className="metric-label"), html.Div(f"{bartlett[0]:.1f}" if bartlett else "—", className="metric-value")]), className="metric-card"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Bartlett p-value", className="metric-label"), html.Div(f"{bartlett[1]:.4f}" if bartlett else "—", className="metric-value")]), className="metric-card"), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("KMO score", className="metric-label"), html.Div(f"{kmo[1]:.3f}" if kmo else "—", className="metric-value")]), className="metric-card"), width=3),
            ], id="s7-test-metrics"),
        ]), className="mb-3"),

        # Step 2: Scree plot
        dbc.Card(dbc.CardBody([
            html.H5("Step 2 — Scree plot & number of factors"),
            dbc.Button("Compute scree plot (eigenvalues)", id="s7-scree-btn", color="secondary", n_clicks=0),
            dcc.Graph(id="s7-scree-plot", figure=scree_fig),
            html.Div(id="s7-scree-info"),
        ]), className="mb-3"),

        # Step 3: Run FA
        dbc.Card(dbc.CardBody([
            html.H5("Step 3 — Run factor analysis"),
            dbc.Row([
                dbc.Col([html.Label("Number of factors"), dbc.Input(id="s7-n-factors", type="number", value=3, min=1, max=max_factors)], width=3),
                dbc.Col([html.Label("Rotation"), dcc.Dropdown(
                    id="s7-rotation",
                    options=[{"label": r, "value": r} for r in ["varimax", "promax", "oblimin", "none"]],
                    value="varimax", clearable=False,
                )], width=3),
            ]),
            html.Br(),
            dbc.Button("Run Factor Analysis", id="s7-run-btn", color="primary", n_clicks=0),
            dcc.Loading(html.Div(id="s7-fa-status"), type="circle"),
        ]), className="mb-3") if _check_fa() else dbc.Alert(
            "`factor_analyzer` not installed. Run: pip install factor_analyzer", color="danger"
        ),

        # Step 4: Loadings (shown if FA run)
        dbc.Card(dbc.CardBody([
            html.H5("Step 4 — Factor loadings"),
            dbc.Row([
                dbc.Col([
                    html.Label("Highlight loadings |λ| >="),
                    dcc.Slider(id="s7-load-thresh", min=0.3, max=0.9, step=0.05, value=0.4,
                               marks={0.3:"0.3", 0.4:"0.4", 0.6:"0.6", 0.9:"0.9"}),
                ], width=6),
            ]),
            html.Div(id="s7-loadings-table"),
        ]), className="mb-3") if fa_loadings is not None else html.Div(),

        # Step 5: output mode and apply
        dbc.Card(dbc.CardBody([
            html.H5("Step 5 — Output mode"),
            dcc.RadioItems(
                id="s7-output-mode",
                options=[
                    {"label": "Use factor scores as new features (F1, F2, ...)", "value": "scores"},
                    {"label": "Keep high-loading original features only", "value": "highload"},
                    {"label": "Keep both factors and original features", "value": "both"},
                ],
                value="scores", inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "20px"},
            ),
            html.Div(id="s7-highload-select"),
            html.Br(),
            dbc.Button("Apply factor analysis results", id="s7-apply-btn", color="primary", n_clicks=0),
            html.Div(id="s7-apply-status", className="mt-2"),
        ]), className="mb-3") if fa_loadings is not None else html.Div(),
    ])


@callback(
    Output("s7-skip-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s7-skip", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def handle_skip(skip_val, state):
    state = dict(state or {})
    if "skip" in (skip_val or []):
        X_train = server_store.get_df("X_train")
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])] if X_train is not None else []
        state["factor_done"] = True
        state["factor_features"] = num_cols
        return dbc.Alert("Skipped. Proceed to Stage 8 — Combination Testing.", color="info"), state
    return html.Div(), state


@callback(
    Output("s7-tests-status", "children"),
    Output("s7-test-metrics", "children"),
    Input("s7-tests-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_tests(n_clicks):
    if not n_clicks or not _check_fa():
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data.", color="danger"), no_update
    try:
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        X_num = X_train[num_cols].dropna()
        chi2, p = calculate_bartlett_sphericity(X_num)
        kmo_all, kmo_model = calculate_kmo(X_num)
        server_store.set_val("fa_bartlett", (chi2, p))
        server_store.set_val("fa_kmo", (kmo_all, kmo_model))
        metrics = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Bartlett χ²", className="metric-label"), html.Div(f"{chi2:.1f}", className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Bartlett p-value", className="metric-label"), html.Div(f"{p:.4f}", className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("KMO score", className="metric-label"), html.Div(f"{kmo_model:.3f}", className="metric-value")]), className="metric-card"), width=3),
        ])
        msg = []
        if p < 0.05:
            msg.append(dbc.Alert("Bartlett test significant — data suitable for FA.", color="success", className="py-2"))
        else:
            msg.append(dbc.Alert("Bartlett test not significant — variables may not be correlated enough.", color="warning", className="py-2"))
        if kmo_model >= 0.6:
            msg.append(dbc.Alert(f"KMO={kmo_model:.3f} — adequate for FA.", color="success", className="py-2"))
        else:
            msg.append(dbc.Alert(f"KMO={kmo_model:.3f} — inadequate (< 0.6).", color="warning", className="py-2"))
        return html.Div(msg), metrics
    except Exception as e:
        return dbc.Alert(f"Test error: {e}", color="danger"), no_update


@callback(
    Output("s7-scree-plot", "figure"),
    Output("s7-scree-info", "children"),
    Input("s7-scree-btn", "n_clicks"),
    prevent_initial_call=True,
)
def compute_scree(n_clicks):
    if not n_clicks or not _check_fa():
        return no_update, no_update
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return no_update, no_update
    try:
        from factor_analyzer import FactorAnalyzer
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        X_num = X_train[num_cols].dropna()
        max_factors = min(len(num_cols) - 1, 15)
        fa_scree = FactorAnalyzer(n_factors=max_factors, rotation=None)
        fa_scree.fit(X_num)
        ev, _ = fa_scree.get_eigenvalues()
        ev_list = ev[:max_factors].tolist()
        server_store.set_val("fa_eigenvalues", ev_list)
        n_above_1 = sum(1 for e in ev_list if e >= 1.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(ev_list)+1)), y=ev_list,
            mode="lines+markers", marker=dict(size=8), line=dict(color="#2E75B6"),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="Kaiser criterion (eigenvalue=1)")
        fig.update_layout(title="Scree Plot", xaxis_title="Factor", yaxis_title="Eigenvalue",
                          height=350, margin=dict(t=50, b=30))
        return fig, dbc.Alert(f"{n_above_1} factors have eigenvalue >= 1 (Kaiser criterion).", color="info", className="py-2")
    except Exception as e:
        return no_update, dbc.Alert(f"Scree error: {e}", color="danger")


@callback(
    Output("s7-fa-status", "children"),
    Input("s7-run-btn", "n_clicks"),
    State("s7-n-factors", "value"),
    State("s7-rotation", "value"),
    prevent_initial_call=True,
)
def run_fa(n_clicks, n_factors, rotation):
    if not n_clicks or not _check_fa():
        return no_update
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if X_train is None:
        return dbc.Alert("No data.", color="danger")
    try:
        from factor_analyzer import FactorAnalyzer
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        X_num = X_train[num_cols].dropna()
        rotation_arg = None if rotation == "none" else rotation
        n_f = int(n_factors or 3)
        fa = FactorAnalyzer(n_factors=n_f, rotation=rotation_arg)
        fa.fit(X_num)
        loadings = pd.DataFrame(
            fa.loadings_, index=num_cols,
            columns=[f"F{i+1}" for i in range(n_f)],
        ).round(3)
        variance = pd.DataFrame(
            fa.get_factor_variance(),
            index=["SS Loadings", "Proportion Var", "Cumulative Var"],
            columns=[f"F{i+1}" for i in range(n_f)],
        ).round(3)
        scores_tr = fa.transform(X_num)
        mean_vals = X_num.mean()
        scores_te = fa.transform(X_test[num_cols].fillna(mean_vals))

        server_store.set_val("fa_model", fa)
        server_store.set_val("fa_loadings", loadings)
        server_store.set_val("fa_variance", variance)
        server_store.set_val("fa_scores_tr", scores_tr)
        server_store.set_val("fa_scores_te", scores_te)
        server_store.set_val("fa_num_cols", num_cols)

        return dbc.Alert(f"Factor analysis complete ({n_f} factors, {rotation} rotation). Reload page to see loadings.", color="success")
    except Exception as e:
        return dbc.Alert(f"FA error: {e}", color="danger")


@callback(
    Output("s7-loadings-table", "children"),
    Input("s7-load-thresh", "value"),
    prevent_initial_call=True,
)
def update_loadings_table(thresh):
    loadings = server_store.get_val("fa_loadings")
    if loadings is None:
        return dbc.Alert("Run Factor Analysis first.", color="info")

    thresh = thresh or 0.4
    rows = []
    for idx, row in loadings.iterrows():
        cells = [html.Td(idx)]
        for col in loadings.columns:
            val = row[col]
            style = {"backgroundColor": "#d6e4f7", "fontWeight": "bold"} if abs(val) >= thresh else {}
            cells.append(html.Td(f"{val:.3f}", style=style))
        rows.append(html.Tr(cells))

    header = html.Tr([html.Th("Feature")] + [html.Th(c) for c in loadings.columns])
    table = dbc.Table([html.Thead(header), html.Tbody(rows)], bordered=True, hover=True, size="sm",
                      style={"fontSize": "0.82rem"})

    fa_variance = server_store.get_val("fa_variance")
    var_section = html.Div()
    if fa_variance is not None:
        var_section = html.Div([
            html.H6("Variance explained:"),
            dbc.Table.from_dataframe(fa_variance.reset_index().rename(columns={"index": "Metric"}),
                                     striped=True, size="sm"),
        ], className="mt-3")

    return html.Div([table, var_section])


@callback(
    Output("s7-highload-select", "children"),
    Input("s7-output-mode", "value"),
    Input("s7-load-thresh", "value"),
    prevent_initial_call=True,
)
def update_highload_select(mode, thresh):
    loadings = server_store.get_val("fa_loadings")
    if loadings is None or mode not in ("highload", "both"):
        return html.Div(dcc.Dropdown(id="s7-highload-feats", options=[], multi=True), style={"display": "none"})
    thresh = thresh or 0.4
    high_feats = loadings.abs().max(axis=1)
    high_feats = high_feats[high_feats >= thresh].index.tolist()
    num_cols = server_store.get_val("fa_num_cols") or list(loadings.index)
    return html.Div([
        html.Label(f"High-loading features (|λ| >= {thresh}):"),
        dcc.Dropdown(
            id="s7-highload-feats",
            options=[{"label": c, "value": c} for c in num_cols],
            value=high_feats,
            multi=True,
        ),
    ], className="mt-2")


@callback(
    Output("s7-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s7-apply-btn", "n_clicks"),
    State("s7-output-mode", "value"),
    State("s7-highload-feats", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_fa(n_clicks, output_mode, highload_feats, state):
    if not n_clicks:
        return no_update, no_update

    loadings = server_store.get_val("fa_loadings")
    scores_tr = server_store.get_val("fa_scores_tr")
    scores_te = server_store.get_val("fa_scores_te")
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    if loadings is None or scores_tr is None:
        return dbc.Alert("Run Factor Analysis first.", color="warning"), no_update

    factor_cols = list(loadings.columns)
    fa_tr_df = pd.DataFrame(scores_tr, index=X_train.index[:len(scores_tr)], columns=factor_cols)
    fa_te_df = pd.DataFrame(scores_te, index=X_test.index[:len(scores_te)], columns=factor_cols)

    if output_mode == "scores":
        X_tr_out = fa_tr_df
        X_te_out = fa_te_df
        out_features = factor_cols
    elif output_mode == "highload":
        feats = list(highload_feats or [])
        avail = [f for f in feats if f in X_train.columns]
        X_tr_out = X_train[avail]
        X_te_out = X_test[avail]
        out_features = avail
    else:  # both
        feats = list(highload_feats or [])
        avail = [f for f in feats if f in X_train.columns]
        X_tr_out = pd.concat([fa_tr_df, X_train[avail].reset_index(drop=True)], axis=1)
        X_te_out = pd.concat([fa_te_df, X_test[avail].reset_index(drop=True)], axis=1)
        out_features = factor_cols + avail

    X_tr_out = X_tr_out.loc[:, ~X_tr_out.columns.duplicated()]
    X_te_out = X_te_out.loc[:, ~X_te_out.columns.duplicated()]
    out_features = list(dict.fromkeys(out_features))

    server_store.set_df("X_train", X_tr_out)
    server_store.set_df("X_test", X_te_out)

    state = dict(state or {})
    state["factor_done"] = True
    state["factor_features"] = out_features

    return dbc.Alert(
        f"Factor analysis applied — {X_tr_out.shape[1]} features entering Stage 8.",
        color="success",
    ), state
