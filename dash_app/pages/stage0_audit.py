"""
Stage 0 — Data Audit
Upload dataset, inspect quality, assign column types, drop unwanted columns.
"""
import base64, io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store

# ── helpers ───────────────────────────────────────────────────────────────────

def _decode_upload(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
    else:
        return pd.read_excel(io.BytesIO(decoded))


def _build_editor_rows(df: pd.DataFrame, target: str, prior: dict) -> list:
    feature_cols = [c for c in df.columns if c != target]
    auto_numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in feature_cols:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        non_null = pd.to_numeric(s.dropna(), errors="coerce").dropna() if is_num else s.dropna()
        miss_pct = round(s.isnull().sum() / max(len(s), 1) * 100, 1)
        n_unique = int(s.nunique())
        variance = round(float(non_null.var()), 4) if is_num and len(non_null) > 1 else None
        zero_pct = round(float((non_null == 0).sum() / max(len(non_null), 1) * 100), 1) \
                   if is_num and len(non_null) > 0 else None
        skew_val = round(float(non_null.skew()), 3) if is_num and len(non_null) > 2 else None
        default_type = prior.get(col, {}).get("Type",
                       "numeric" if col in auto_numeric else "categorical")
        default_drop = prior.get(col, {}).get("Drop", False)
        rows.append({
            "Column": col,
            "Type": default_type,
            "Drop": default_drop,
            "Missing%": miss_pct,
            "Zero%": zero_pct if zero_pct is not None else "",
            "Variance": variance if variance is not None else "",
            "Skewness": skew_val if skew_val is not None else "",
            "Unique": n_unique,
            "Dtype": str(s.dtype),
        })
    return rows


def _col_defs() -> list:
    return [
        {"field": "Column",   "editable": False, "width": 160, "pinned": "left"},
        {
            "field": "Type", "editable": True, "width": 130,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": ["numeric", "categorical", "ordinal"]},
        },
        {
            "field": "Drop", "editable": True, "width": 80,
            "cellRenderer": "agCheckboxCellRenderer",
            "cellEditor": "agCheckboxCellEditor",
        },
        {"field": "Missing%",  "editable": False, "width": 100, "headerName": "Missing %"},
        {"field": "Zero%",     "editable": False, "width": 90,  "headerName": "Zero %"},
        {"field": "Variance",  "editable": False, "width": 100},
        {"field": "Skewness",  "editable": False, "width": 100},
        {"field": "Unique",    "editable": False, "width": 80},
        {"field": "Dtype",     "editable": False, "width": 100},
    ]


# ── layout ────────────────────────────────────────────────────────────────────

def layout(state: dict) -> html.Div:
    raw_df = server_store.get_df("raw_df")
    has_data = raw_df is not None

    upload_section = dbc.Card(
        dbc.CardBody([
            html.H5("Upload dataset (CSV or Excel)", className="card-title"),
            dcc.Upload(
                id="s0-upload",
                children=html.Div([
                    html.I(className="bi bi-cloud-upload me-2"),
                    "Drag & drop or ",
                    html.A("click to select", style={"color": "#3a6df0"}),
                ]),
                accept=".csv,.xlsx,.xls",
                className="upload-area",
                style={"cursor": "pointer"},
            ),
            html.Div(id="s0-upload-status"),
        ]),
        className="mb-3",
    )

    if not has_data:
        return html.Div([
            html.Div([
                html.H2("Stage 0 — Data Audit"),
                html.P("Upload your dataset and review its quality before any processing."),
            ], className="stage-header"),
            upload_section,
        ])

    df = raw_df
    target = state.get("target_col") or df.columns[-1]
    prior = state.get("_col_editor_prior", {})
    editor_rows = _build_editor_rows(df, target, prior)
    # Filter out columns that were previously dropped so they don't reappear on re-render
    if prior:
        editor_rows = [r for r in editor_rows if not prior.get(r["Column"], {}).get("Drop", False)]

    # Class distribution chart
    class_counts = df[target].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(df) * 100).round(1)
    class_fig = px.bar(
        class_counts, x="Class", y="Count", text="Percentage", color="Class",
        title=f"Class distribution — '{target}'",
    )
    class_fig.update_traces(texttemplate="%{text}%", textposition="outside")
    class_fig.update_layout(showlegend=False, height=300, margin=dict(t=50, b=20))

    # Missing values chart
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Column": miss.index, "Missing %": miss_pct.values}).query("`Missing %` > 0").sort_values("Missing %", ascending=False)
    if not miss_df.empty:
        miss_fig = px.bar(miss_df, x="Column", y="Missing %", color="Missing %",
                          color_continuous_scale="Reds", title="Missing % per column")
        miss_fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="40% threshold")
        miss_fig.update_layout(height=280, margin=dict(t=50, b=20))
        missing_section = dbc.Card(dbc.CardBody([
            html.H5(f"Missing values — {len(miss_df)} columns affected"),
            dcc.Graph(figure=miss_fig),
        ]), className="mb-3")
    else:
        missing_section = dbc.Alert("No missing values found.", color="success", className="mb-3")

    return html.Div([
        html.Div([
            html.H2("Stage 0 — Data Audit"),
            html.P("Upload your dataset and review its quality before any processing."),
        ], className="stage-header"),

        upload_section,
        html.Div(id="s0-upload-status"),

        # Dataset summary metrics
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Rows", className="metric-label"), html.Div(df.shape[0], className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Columns", className="metric-label"), html.Div(df.shape[1], className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Duplicates", className="metric-label"), html.Div(int(df.duplicated().sum()), className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Missing cells", className="metric-label"), html.Div(int(df.isnull().sum().sum()), className="metric-value")]), className="metric-card"), width=3),
        ], className="mb-3"),

        # Target selection
        dbc.Card(dbc.CardBody([
            html.H5("Target (label) column"),
            dcc.Dropdown(
                id="s0-target-select",
                options=[{"label": c, "value": c} for c in df.columns],
                value=target,
                clearable=False,
                style={"maxWidth": "400px"},
            ),
        ]), className="mb-3"),

        # Column editor
        dbc.Card(dbc.CardBody([
            html.H5("Column editor"),
            html.P("Edit Type inline. Tick Drop to remove before processing. Click Apply when done.",
                   style={"color": "#6b7280", "fontSize": "0.85rem"}),
            dag.AgGrid(
                id="s0-col-grid",
                rowData=editor_rows,
                columnDefs=_col_defs(),
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"rowSelection": "multiple", "animateRows": True},
                style={"height": min(500, max(280, len(editor_rows) * 36 + 56)), "width": "100%"},
                className="ag-theme-alpine",
            ),
            html.Br(),
            # Live summary metrics (updated reactively)
            html.Div(id="s0-live-summary"),
            html.Br(),
            dbc.Button("Apply changes", id="s0-apply-btn", color="primary", n_clicks=0),
            html.Div(id="s0-apply-status", className="mt-2"),
        ]), className="mb-3"),

        # Quick-flag section
        dbc.Card(dbc.CardBody([
            html.H5("Quick-flag sweep"),
            html.P("Pre-tick the Drop column for problem columns. Then click Apply changes above.",
                   style={"color": "#6b7280", "fontSize": "0.85rem"}),
            dbc.Row([
                dbc.Col([html.Label("Missing % above"), dcc.Slider(id="s0-sw-miss", min=0, max=100, step=5, value=50, marks={0:"0",50:"50",100:"100"})], width=4),
                dbc.Col([html.Label("Variance below"), dcc.Slider(id="s0-sw-var", min=0.0, max=1.0, step=0.001, value=0.01, marks={0:"0",0.5:"0.5",1:"1"})], width=4),
                dbc.Col([html.Label("Zero % above"), dcc.Slider(id="s0-sw-zero", min=0, max=100, step=5, value=90, marks={0:"0",50:"50",100:"100"})], width=4),
            ], className="mb-2"),
            html.Div(id="s0-flag-summary"),
            dbc.Row([
                dbc.Col(dbc.Button("Tick all-null", id="s0-btn-null", color="secondary", size="sm", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("Tick all-zero", id="s0-btn-zero", color="secondary", size="sm", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("Tick low-var",  id="s0-btn-var",  color="secondary", size="sm", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("Tick ALL flagged", id="s0-btn-all", color="warning", size="sm", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("Clear all Drop ticks", id="s0-btn-clear", color="light", size="sm", n_clicks=0), width="auto"),
            ], className="g-2"),
        ]), className="mb-3"),

        # Charts
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=class_fig)])), width=6),
            dbc.Col(missing_section, width=6),
        ]),
    ])


# ── callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("s0-upload-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s0-upload", "contents"),
    State("s0-upload", "filename"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, state):
    if contents is None:
        return no_update, no_update
    try:
        df = _decode_upload(contents, filename)
        server_store.set_df("raw_df", df.copy())
        server_store.set_df("df", df.copy())
        if state is None:
            state = {}
        state = dict(state)
        state["raw_df_loaded"] = True
        state["target_col"] = df.columns[-1]
        state["_col_editor_prior"] = {}
        # Auto-detect column types
        feature_cols = [c for c in df.columns if c != df.columns[-1]]
        auto_numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        state["numeric_cols"] = [c for c in feature_cols if c in auto_numeric]
        state["categorical_cols"] = [c for c in feature_cols if c not in auto_numeric]
        state["ordinal_cols"] = []
        alert = dbc.Alert(
            f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from '{filename}'",
            color="success", dismissable=True,
        )
        return alert, state
    except Exception as e:
        return dbc.Alert(f"Upload error: {e}", color="danger"), no_update


@callback(
    Output("s0-live-summary", "children"),
    Input("s0-col-grid", "rowData"),
    prevent_initial_call=True,
)
def live_summary(row_data):
    if not row_data:
        return ""
    n_drop = sum(1 for r in row_data if r.get("Drop"))
    kept = [r for r in row_data if not r.get("Drop")]
    n_num = sum(1 for r in kept if r.get("Type") == "numeric")
    n_cat = sum(1 for r in kept if r.get("Type") == "categorical")
    n_ord = sum(1 for r in kept if r.get("Type") == "ordinal")
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Queued to drop", className="metric-label"), html.Div(n_drop, className="metric-value", style={"color": "#ef4444" if n_drop else "inherit"})]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Numeric (kept)", className="metric-label"), html.Div(n_num, className="metric-value")]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Categorical (kept)", className="metric-label"), html.Div(n_cat, className="metric-value")]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Ordinal (kept)", className="metric-label"), html.Div(n_ord, className="metric-value")]), className="metric-card"), width=3),
    ])


@callback(
    Output("s0-apply-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Output("s0-col-grid", "rowData", allow_duplicate=True),
    Input("s0-apply-btn", "n_clicks"),
    State("s0-col-grid", "rowData"),
    State("s0-target-select", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def apply_changes(n_clicks, row_data, target, state):
    if not n_clicks or not row_data:
        return no_update, no_update, no_update
    raw_df = server_store.get_df("raw_df")
    if raw_df is None:
        return dbc.Alert("No data loaded.", color="danger"), no_update, no_update

    state = dict(state or {})
    kept = [r for r in row_data if not r.get("Drop")]
    drops = [r["Column"] for r in row_data if r.get("Drop")]

    numeric_cols = [r["Column"] for r in kept if r.get("Type") == "numeric"]
    categorical_cols = [r["Column"] for r in kept if r.get("Type") == "categorical"]
    ordinal_cols = [r["Column"] for r in kept if r.get("Type") == "ordinal"]

    clean_df = raw_df.drop(columns=drops, errors="ignore")
    server_store.set_df("df", clean_df)

    # Save prior for the editor so it persists on re-render
    prior = {r["Column"]: {"Type": r.get("Type", "numeric"), "Drop": r.get("Drop", False)} for r in row_data}
    state["_col_editor_prior"] = prior
    state["target_col"] = target
    state["numeric_cols"] = numeric_cols
    state["categorical_cols"] = categorical_cols
    state["ordinal_cols"] = ordinal_cols
    state["dropped_cols_audit"] = drops
    state["raw_df_loaded"] = True

    # Return only kept rows (dropped columns removed from the grid)
    new_grid_rows = [{**r, "Drop": False} for r in kept]

    alert = dbc.Alert(
        f"Applied — keeping {len(kept)} columns ({len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical, {len(ordinal_cols)} ordinal). "
        f"Dropped {len(drops)}.",
        color="success", dismissable=True,
    )
    return alert, state, new_grid_rows


@callback(
    Output("s0-col-grid", "rowData", allow_duplicate=True),
    Output("s0-flag-summary", "children"),
    Input("s0-sw-miss",   "value"),
    Input("s0-sw-var",    "value"),
    Input("s0-sw-zero",   "value"),
    Input("s0-btn-null",  "n_clicks"),
    Input("s0-btn-zero",  "n_clicks"),
    Input("s0-btn-var",   "n_clicks"),
    Input("s0-btn-all",   "n_clicks"),
    Input("s0-btn-clear", "n_clicks"),
    State("s0-col-grid",  "rowData"),
    State("s0-target-select", "value"),
    prevent_initial_call=True,
)
def flag_sweep(miss_thresh, var_thresh, zero_thresh,
               n_null, n_zero, n_var, n_all, n_clear,
               row_data, target):
    raw_df = server_store.get_df("raw_df")
    if raw_df is None or not row_data:
        return no_update, no_update

    triggered = ctx.triggered_id

    # Compute flags from raw_df
    feature_cols = [c for c in raw_df.columns if c != (target or raw_df.columns[-1])]
    flagged_null, flagged_zero_c, flagged_var_c, flagged_miss, flagged_zp = [], [], [], [], []
    for col in feature_cols:
        s = raw_df[col]
        n_miss = s.isnull().sum()
        miss_pct = n_miss / max(len(s), 1) * 100
        is_num = pd.api.types.is_numeric_dtype(s)
        non_null = pd.to_numeric(s.dropna(), errors="coerce").dropna() if is_num else pd.Series([], dtype=float)
        variance = float(non_null.var()) if len(non_null) > 1 else None
        zero_pct = float((non_null == 0).sum() / max(len(non_null), 1) * 100) if len(non_null) > 0 else 0.0

        if n_miss == len(s): flagged_null.append(col)
        if is_num and len(non_null) > 0 and float(non_null.abs().max()) == 0: flagged_zero_c.append(col)
        if variance is not None and variance == 0.0: flagged_zero_c.append(col)
        if variance is not None and variance < (var_thresh or 0.01): flagged_var_c.append(col)
        if miss_pct >= (miss_thresh or 50): flagged_miss.append(col)
        if zero_pct >= (zero_thresh or 90): flagged_zp.append(col)

    flagged_zero_c = list(set(flagged_zero_c))
    flagged_var_c = list(set(flagged_var_c))
    flagged_all = list(set(flagged_null + flagged_zero_c + flagged_var_c + flagged_miss + flagged_zp))

    # Determine which cols to tick
    cols_to_tick = None
    if triggered == "s0-btn-null":  cols_to_tick = set(flagged_null)
    elif triggered == "s0-btn-zero": cols_to_tick = set(flagged_zero_c)
    elif triggered == "s0-btn-var":  cols_to_tick = set(flagged_var_c)
    elif triggered == "s0-btn-all":  cols_to_tick = set(flagged_all)
    elif triggered == "s0-btn-clear": cols_to_tick = set()

    new_rows = list(row_data)
    if cols_to_tick is not None:
        for r in new_rows:
            if triggered == "s0-btn-clear":
                r["Drop"] = False
            elif r["Column"] in cols_to_tick:
                r["Drop"] = True

    summary = dbc.Row([
        dbc.Col(dbc.Badge(f"All-null: {len(flagged_null)}", color="danger", className="me-1")),
        dbc.Col(dbc.Badge(f"All-zero/const: {len(flagged_zero_c)}", color="warning", className="me-1")),
        dbc.Col(dbc.Badge(f"Low-var: {len(flagged_var_c)}", color="info", className="me-1")),
        dbc.Col(dbc.Badge(f"High-miss: {len(flagged_miss)}", color="secondary", className="me-1")),
        dbc.Col(dbc.Badge(f"High-zero%: {len(flagged_zp)}", color="secondary", className="me-1")),
    ], className="mb-2")

    return new_rows, summary


@callback(
    Output("s0-target-select", "options"),
    Output("s0-target-select", "value"),
    Input("app-state", "data"),
)
def refresh_target_options(state):
    raw_df = server_store.get_df("raw_df")
    if raw_df is None:
        return [], None
    options = [{"label": c, "value": c} for c in raw_df.columns]
    value = (state or {}).get("target_col") or raw_df.columns[-1]
    return options, value
