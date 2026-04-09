"""
Variable Mapping — pre-Stage 0
Upload raw data + questionnaire file; shows every dataset column mapped to
its QNR question code, question text, type and answer options.
"""
import base64, io
import pandas as pd

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store
from utils.qnr_parser import parse_questionnaire
from utils.col_mapper import build_codebook, group_columns


# ── helpers ───────────────────────────────────────────────────────────────────

def _decode_df(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
    return pd.read_excel(io.BytesIO(decoded))


def _decode_qnr(contents: str, filename: str, tmp_path: str) -> list:
    """Write QNR bytes to a temp file, parse, return question list."""
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    with open(tmp_path, "wb") as f:
        f.write(decoded)
    return parse_questionnaire(tmp_path)


def _build_mapping_rows(df: pd.DataFrame, questions: list) -> list:
    """
    Return one row per dataset column with its matched QNR fields.
    Also appends QNR questions that have no matching column.
    """
    col_groups = group_columns(df)
    q_map = {q["code"].upper(): q for q in questions}

    rows = []
    matched_codes = set()

    for col in df.columns:
        # Find best matching QNR code
        import re
        m = re.match(r'^([A-Za-z]\d+[a-z]?)', col)
        prefix = m.group(1).upper() if m else None
        q = q_map.get(prefix) if prefix else None

        if q:
            matched_codes.add(prefix)
            options_str = " | ".join(q.get("options", [])[:20])  # cap display
            q_type = q.get("q_type", "")
            scale = ""
            if q.get("scale_points"):
                scale = f"{q['scale_points']}-pt  [{q.get('scale_low','')} — {q.get('scale_high','')}]"
            rows.append({
                "Dataset Column": col,
                "QNR Code":       q["code"],
                "Question":       q["question"],
                "Q Type":         q_type,
                "Scale":          scale,
                "Options":        options_str,
                "Status":         "Matched",
            })
        else:
            rows.append({
                "Dataset Column": col,
                "QNR Code":       "",
                "Question":       "",
                "Q Type":         "",
                "Scale":          "",
                "Options":        "",
                "Status":         "No QNR match",
            })

    # QNR questions with no dataset column
    for q in questions:
        code = q["code"].upper()
        if code not in matched_codes and code not in col_groups:
            options_str = " | ".join(q.get("options", [])[:20])
            scale = ""
            if q.get("scale_points"):
                scale = f"{q['scale_points']}-pt  [{q.get('scale_low','')} — {q.get('scale_high','')}]"
            rows.append({
                "Dataset Column": "",
                "QNR Code":       q["code"],
                "Question":       q["question"],
                "Q Type":         q.get("q_type", ""),
                "Scale":          scale,
                "Options":        options_str,
                "Status":         "QNR only",
            })

    return rows


def _col_defs() -> list:
    status_styles = [
        {"condition": "params.value === 'Matched'",      "style": {"color": "#16a34a", "fontWeight": "600"}},
        {"condition": "params.value === 'No QNR match'", "style": {"color": "#d97706"}},
        {"condition": "params.value === 'QNR only'",     "style": {"color": "#6b7280", "fontStyle": "italic"}},
    ]
    return [
        {"field": "Dataset Column", "pinned": "left",  "width": 170, "editable": False},
        {"field": "QNR Code",       "width": 110,      "editable": False},
        {"field": "Question",       "flex": 3,         "editable": False,
         "wrapText": True, "autoHeight": True},
        {"field": "Q Type",         "width": 110,      "editable": False},
        {"field": "Scale",          "width": 200,      "editable": False},
        {"field": "Options",        "flex": 2,         "editable": False,
         "wrapText": True, "autoHeight": True},
        {
            "field": "Status", "width": 130, "editable": False,
            "cellStyle": {"styleConditions": status_styles},
        },
    ]


def _summary_badges(rows: list) -> html.Div:
    matched   = sum(1 for r in rows if r["Status"] == "Matched")
    no_match  = sum(1 for r in rows if r["Status"] == "No QNR match")
    qnr_only  = sum(1 for r in rows if r["Status"] == "QNR only")
    data_cols = sum(1 for r in rows if r["Dataset Column"])
    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Data Columns", className="metric-label"),
            html.Div(data_cols, className="metric-value"),
        ]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Matched to QNR", className="metric-label"),
            html.Div(matched, className="metric-value", style={"color": "#16a34a"}),
        ]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("No QNR match", className="metric-label"),
            html.Div(no_match, className="metric-value", style={"color": "#d97706"}),
        ]), className="metric-card"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("QNR only (no data col)", className="metric-label"),
            html.Div(qnr_only, className="metric-value", style={"color": "#6b7280"}),
        ]), className="metric-card"), width=3),
    ], className="mb-3")


# ── layout ────────────────────────────────────────────────────────────────────

def layout(state: dict) -> html.Div:
    raw_df    = server_store.get_df("raw_df")
    questions = server_store.get_val("qnr_questions", [])
    has_df    = raw_df is not None
    has_qnr   = bool(questions)

    header = html.Div([
        html.H2("Variable Mapping"),
        html.P(
            "Upload your raw data file and questionnaire to see how every "
            "dataset column maps to its QNR question, type and answer options.",
            style={"color": "#6b7280"},
        ),
    ], className="stage-header")

    upload_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6([html.I(className="bi bi-table me-2"), "Raw data (CSV / Excel)"],
                    className="card-title"),
            dcc.Upload(
                id="vm-upload-data",
                children=html.Div([
                    html.I(className="bi bi-cloud-upload me-2"),
                    "Drag & drop or ", html.A("browse", style={"color": "#3a6df0"}),
                ]),
                accept=".csv,.xlsx,.xls",
                className="upload-area",
                style={"cursor": "pointer"},
            ),
            html.Div(id="vm-data-status", className="mt-2"),
        ])), width=6),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6([html.I(className="bi bi-file-earmark-text me-2"),
                     "Questionnaire (docx / pdf / xlsx / txt)"],
                    className="card-title"),
            dcc.Upload(
                id="vm-upload-qnr",
                children=html.Div([
                    html.I(className="bi bi-cloud-upload me-2"),
                    "Drag & drop or ", html.A("browse", style={"color": "#3a6df0"}),
                ]),
                accept=".docx,.pdf,.xlsx,.xls,.txt",
                className="upload-area",
                style={"cursor": "pointer"},
            ),
            html.Div(id="vm-qnr-status", className="mt-2"),
        ])), width=6),
    ], className="mb-3")

    # Mapping table (only shown when both files are loaded)
    if has_df and has_qnr:
        rows = _build_mapping_rows(raw_df, questions)
        grid = dag.AgGrid(
            id="vm-grid",
            rowData=rows,
            columnDefs=_col_defs(),
            defaultColDef={
                "resizable": True, "sortable": True, "filter": True,
                "wrapHeaderText": True, "autoHeaderHeight": True,
            },
            dashGridOptions={"animateRows": True, "rowSelection": "single"},
            style={"height": "600px", "width": "100%"},
            className="ag-theme-alpine",
        )
        mapping_section = html.Div([
            _summary_badges(rows),
            dbc.Card(dbc.CardBody([
                html.H5("Column — QNR mapping table"),
                html.P(
                    "Green = matched | Amber = dataset column with no QNR entry | "
                    "Grey italic = QNR question not found in dataset",
                    style={"color": "#6b7280", "fontSize": "0.82rem"},
                ),
                grid,
            ])),
        ])
    elif has_df and not has_qnr:
        mapping_section = dbc.Alert(
            "Raw data loaded. Upload the questionnaire file to see the mapping.",
            color="info",
        )
    elif not has_df and has_qnr:
        mapping_section = dbc.Alert(
            "Questionnaire loaded. Upload the raw data file to see the mapping.",
            color="info",
        )
    else:
        mapping_section = dbc.Alert(
            "Upload both files above to generate the variable mapping.",
            color="secondary",
        )

    return html.Div([header, upload_row, html.Div(id="vm-mapping-area", children=mapping_section)])


# ── callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("vm-data-status",   "children"),
    Output("app-state",        "data", allow_duplicate=True),
    Input("vm-upload-data",    "contents"),
    State("vm-upload-data",    "filename"),
    State("app-state",         "data"),
    prevent_initial_call=True,
)
def upload_data(contents, filename, state):
    if not contents:
        return no_update, no_update
    try:
        df = _decode_df(contents, filename)
        server_store.set_df("raw_df", df.copy())
        server_store.set_df("df",     df.copy())
        state = dict(state or {})
        state["raw_df_loaded"] = True
        state["target_col"]    = df.columns[-1]
        import numpy as np
        feature_cols = [c for c in df.columns if c != df.columns[-1]]
        auto_num = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        state["numeric_cols"]     = [c for c in feature_cols if c in auto_num]
        state["categorical_cols"] = [c for c in feature_cols if c not in auto_num]
        state["ordinal_cols"]     = []
        # Refresh mapping if QNR already loaded
        questions = server_store.get_val("qnr_questions", [])
        if questions:
            state["mapping_done"] = True
        return (
            dbc.Alert(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from '{filename}'",
                      color="success", dismissable=True),
            state,
        )
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger"), no_update


@callback(
    Output("vm-qnr-status",  "children"),
    Output("app-state",      "data", allow_duplicate=True),
    Input("vm-upload-qnr",   "contents"),
    State("vm-upload-qnr",   "filename"),
    State("app-state",       "data"),
    prevent_initial_call=True,
)
def upload_qnr(contents, filename, state):
    if not contents:
        return no_update, no_update
    try:
        import tempfile, pathlib
        ext = pathlib.Path(filename).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        questions = _decode_qnr(contents, filename, tmp.name)
        os.unlink(tmp.name)
        server_store.set_val("qnr_questions", questions)
        state = dict(state or {})
        raw_df = server_store.get_df("raw_df")
        if raw_df is not None:
            state["mapping_done"] = True
        return (
            dbc.Alert(f"Parsed {len(questions)} questions from '{filename}'",
                      color="success", dismissable=True),
            state,
        )
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger"), no_update


@callback(
    Output("vm-mapping-area", "children"),
    Input("app-state",        "data"),
    prevent_initial_call=True,
)
def refresh_mapping(state):
    raw_df    = server_store.get_df("raw_df")
    questions = server_store.get_val("qnr_questions", [])
    has_df    = raw_df is not None
    has_qnr   = bool(questions)

    if has_df and has_qnr:
        rows = _build_mapping_rows(raw_df, questions)
        grid = dag.AgGrid(
            id="vm-grid",
            rowData=rows,
            columnDefs=_col_defs(),
            defaultColDef={
                "resizable": True, "sortable": True, "filter": True,
                "wrapHeaderText": True, "autoHeaderHeight": True,
            },
            dashGridOptions={"animateRows": True, "rowSelection": "single"},
            style={"height": "600px", "width": "100%"},
            className="ag-theme-alpine",
        )
        return html.Div([
            _summary_badges(rows),
            dbc.Card(dbc.CardBody([
                html.H5("Column — QNR mapping table"),
                html.P(
                    "Green = matched | Amber = dataset column with no QNR entry | "
                    "Grey italic = QNR question not found in dataset",
                    style={"color": "#6b7280", "fontSize": "0.82rem"},
                ),
                grid,
            ])),
        ])
    elif has_df:
        return dbc.Alert("Raw data loaded. Upload the questionnaire file to see the mapping.", color="info")
    elif has_qnr:
        return dbc.Alert("Questionnaire loaded. Upload the raw data file to see the mapping.", color="info")
    return dbc.Alert("Upload both files above to generate the variable mapping.", color="secondary")
