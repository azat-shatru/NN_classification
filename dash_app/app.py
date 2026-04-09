"""
NN Pipeline — Dash entry point
Run with:  python dash_app/app.py
or:        cd dash_app && python app.py
"""
import sys, os
# Ensure dash_app directory is on the path so relative imports work
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Import all page modules — this registers their callbacks
from pages import (
    var_mapping,
    stage0_audit,
    stage1_missing,
    stage2_outliers,
    stage2b_viz,
    stage3_encoding,
    stage4_scaling,
    stage5_correlation,
    stage6_rf,
    stage7_factor,
    stage8_combtest,
    stage9_nn,
)

# ── Default app state (stored in dcc.Store as JSON) ───────────────────────────
DEFAULT_STATE = {
    "mapping_done": False,
    "target_col": None,
    "numeric_cols": [],
    "categorical_cols": [],
    "ordinal_cols": [],
    "df_imputed": False,
    "df_outliers_done": False,
    "df_encoded": False,
    "df_scaled": False,
    "corr_done": False,
    "rf_done": False,
    "factor_done": False,
    "nn_trained": False,
    "selected_features": None,
    "dropped_cols_audit": [],
    "enc_strategies": {},
    "ord_orders": {},
    "rf_selected_features": [],
    "factor_features": [],
    # Extra flags
    "raw_df_loaded": False,
    "split_done": False,
    "codebook_ready": False,
}

# ── Stage route mapping ────────────────────────────────────────────────────────
STAGES = [
    ("/mapping", "Variable Mapping",             "mapping_done",     var_mapping),
    ("/stage0", "Stage 0 · Data Audit",         "raw_df_loaded",    stage0_audit),
    ("/stage1", "Stage 1 · Missing Values",      "df_imputed",       stage1_missing),
    ("/stage2", "Stage 2 · Outlier Treatment",   "df_outliers_done", stage2_outliers),
    ("/stage2b","Stage 2.5 · Visualisation",     "codebook_ready",   stage2b_viz),
    ("/stage3", "Stage 3 · Encoding",            "df_encoded",       stage3_encoding),
    ("/stage4", "Stage 4 · Scaling",             "df_scaled",        stage4_scaling),
    ("/stage5", "Stage 5 · Correlation Filter",  "corr_done",        stage5_correlation),
    ("/stage6", "Stage 6 · RF Importance",       "rf_done",          stage6_rf),
    ("/stage7", "Stage 7 · Factor Analysis",     "factor_done",      stage7_factor),
    ("/stage8", "Stage 8 · Combination Testing", "selected_features",stage8_combtest),
    ("/stage9", "Stage 9 · Neural Network",      "nn_trained",       stage9_nn),
]

PATH_TO_IDX = {path: i for i, (path, *_) in enumerate(STAGES)}

# ── App init ──────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="NN Pipeline",
)
server = app.server  # expose Flask server for production deployment


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _sidebar_nav(state: dict, pathname: str) -> list:
    links = []
    done_count = 0
    for path, label, flag_key, _ in STAGES:
        flag_val = state.get(flag_key)
        is_done = bool(flag_val) if not isinstance(flag_val, list) else len(flag_val) > 0
        if is_done:
            done_count += 1
        icon = "bi bi-check-circle-fill text-success me-1" if is_done else "bi bi-circle me-1"
        is_active = pathname == path or (pathname in ("/", "") and path == "/mapping")
        link_classes = "sidebar-link" + (" active" if is_active else "") + (" done" if is_done else "")
        links.append(
            html.A(
                [html.I(className=icon), label],
                href=path,
                className=link_classes,
            )
        )

    progress_pct = done_count / len(STAGES)
    links.append(html.Hr(style={"borderColor": "#2d3448", "margin": "12px 0"}))
    links.append(html.Div("Progress", className="sidebar-progress-label"))
    links.append(
        dbc.Progress(
            value=progress_pct * 100,
            color="success",
            style={"height": "6px", "borderRadius": "3px"},
        )
    )
    links.append(
        html.Div(
            f"{done_count} / {len(STAGES)} stages complete",
            style={"color": "#8b9ab1", "fontSize": "0.72rem", "marginTop": "4px"},
        )
    )
    return links


def _sidebar():
    return html.Div(
        [
            html.H4([html.I(className="bi bi-cpu me-2"), "NN Pipeline"]),
            html.Div("Multiclass Classification · PyTorch", className="sidebar-caption"),
            html.Hr(style={"borderColor": "#2d3448", "margin": "12px 0"}),
            html.Div(id="sidebar-nav"),
        ],
        id="sidebar",
        style={
            "width": "220px",
            "minWidth": "220px",
            "backgroundColor": "#1e2130",
            "minHeight": "100vh",
            "padding": "20px 12px",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "overflowY": "auto",
            "zIndex": 1000,
        },
    )


# ── Main layout ───────────────────────────────────────────────────────────────
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="app-state", storage_type="session", data=DEFAULT_STATE),
        _sidebar(),
        html.Div(
            dcc.Loading(
                html.Div(id="page-content"),
                type="circle",
                color="#3a6df0",
            ),
            style={
                "marginLeft": "220px",
                "padding": "28px 32px",
                "minHeight": "100vh",
                "backgroundColor": "#f8f9fa",
            },
        ),
    ]
)


# ── Routing callback ──────────────────────────────────────────────────────────
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("app-state", "data"),
)
def render_page(pathname, state):
    if state is None:
        state = DEFAULT_STATE.copy()

    # Default to variable mapping
    if not pathname or pathname == "/":
        pathname = "/mapping"

    for path, label, flag_key, page_module in STAGES:
        if pathname == path:
            try:
                return page_module.layout(state)
            except Exception as exc:
                return dbc.Alert(
                    [html.Strong("Page error: "), str(exc)],
                    color="danger",
                )

    return dbc.Alert(f"Page not found: {pathname}", color="warning")


# ── Sidebar update callback ───────────────────────────────────────────────────
@callback(
    Output("sidebar-nav", "children"),
    Input("app-state", "data"),
    Input("url", "pathname"),
)
def update_sidebar(state, pathname):
    if state is None:
        state = DEFAULT_STATE.copy()
    if not pathname:
        pathname = "/mapping"
    return _sidebar_nav(state, pathname)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
