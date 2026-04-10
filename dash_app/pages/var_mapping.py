"""
Variable Mapping — pre-Stage 0
Interactive accordion tile view: each questionnaire question is a card;
each dataset column is a draggable tile with type editor, reassign dropdown,
delete button, and hover preview of first 10 values + answer options.
"""
import base64, io, re, tempfile, pathlib
import pandas as pd
import numpy as np

from dash import dcc, html, Input, Output, State, callback, no_update, ctx, ALL
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store
from utils.qnr_parser import parse_questionnaire
from utils.col_mapper import group_columns, suggest_var_type

# ── constants ─────────────────────────────────────────────────────────────────

VAR_TYPE_OPTIONS = [
    {"label": "numeric",      "value": "numeric"},
    {"label": "categorical",  "value": "categorical"},
    {"label": "ordinal",      "value": "ordinal"},
    {"label": "scale_7",      "value": "scale_7"},
    {"label": "scale_5",      "value": "scale_5"},
    {"label": "multi-select", "value": "multi"},
    {"label": "grid",         "value": "grid"},
    {"label": "open-ended",   "value": "open"},
]

DEFAULT_VM_STATE = {"deleted": [], "type_overrides": {}, "reassigned": {}, "option_assignments": {}}

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_id(col: str) -> str:
    """Sanitise column name to a valid HTML element ID."""
    return "vmt-" + re.sub(r"[^A-Za-z0-9]", "-", col)


def _decode_df(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
    return pd.read_excel(io.BytesIO(decoded))


def _decode_qnr(contents: str, filename: str, tmp_path: str) -> list:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    with open(tmp_path, "wb") as f:
        f.write(decoded)
    return parse_questionnaire(tmp_path)


# ── data grouping ─────────────────────────────────────────────────────────────

def _build_groups(df: pd.DataFrame, questions: list, vm_state: dict) -> list:
    """
    Build ordered list of question groups for rendering.
    Each group dict: code, question, matched_cols, possibly_related,
                     extra_cols (reassigned-in), status, options_str, scale.
    """
    reassigned = vm_state.get("reassigned", {})
    q_map = {q["code"].upper(): q for q in questions}
    col_groups = group_columns(df)

    used_cols: set = set()
    groups = []

    for q in questions:
        code = q["code"].upper()
        grp  = col_groups.get(code, {})
        orig = grp.get("cols", [])
        used_cols.update(orig)

        # Cols originally in this group minus those reassigned away
        matched_cols = [c for c in orig if reassigned.get(c, code) == code]

        # Possibly-related: a different col-group whose prefix starts with this
        # code followed by a letter (e.g. A6B → possibly related to A6)
        possibly_related: list = []
        for prefix, g in col_groups.items():
            if prefix == code or prefix in q_map:
                continue
            rest = prefix.upper()[len(code):]
            if prefix.upper().startswith(code) and rest and rest[0].isalpha():
                for col in g["cols"]:
                    if col not in used_cols and reassigned.get(col) is None:
                        possibly_related.append(col)
                        used_cols.add(col)

        # Cols reassigned INTO this group from elsewhere
        extra_cols = [
            c for c, tgt in reassigned.items()
            if tgt == code and c not in matched_cols and c not in possibly_related
        ]

        scale = (
            f"{q['scale_points']}-pt "
            f"[{q.get('scale_low','')} — {q.get('scale_high','')}]"
            if q.get("scale_points") else ""
        )

        groups.append({
            "code":             code,
            "question":         q,
            "matched_cols":     matched_cols,
            "possibly_related": possibly_related,
            "extra_cols":       extra_cols,
            "status":           "matched" if (matched_cols or extra_cols) else "qnr_only",
            "options_str":      " | ".join(q.get("options", [])[:20]),
            "scale":            scale,
        })

    # Dataset columns with no QNR match whatsoever
    unmatched = [
        col
        for prefix, g in col_groups.items()
        if prefix not in q_map
        for col in g["cols"]
        if col not in used_cols and reassigned.get(col) is None
    ]
    if unmatched:
        groups.append({
            "code":             "_UNMATCHED_",
            "question":         None,
            "matched_cols":     unmatched,
            "possibly_related": [],
            "extra_cols":       [],
            "status":           "no_match",
            "options_str":      "",
            "scale":            "",
        })

    return groups


# ── tile renderer ─────────────────────────────────────────────────────────────

def _render_tile(
    col: str,
    df: pd.DataFrame,
    options_str: str,
    scale_str: str,
    current_type: str,
    current_q_code: str,
    all_q_codes: list,
    is_possibly_related: bool = False,
    is_extra: bool = False,
    current_opt_assignment: list = None,
) -> html.Div:
    """Render one variable tile with inline options."""
    tid = _safe_id(col)

    # First 10 non-null values for tooltip
    if col in df.columns:
        vals     = df[col].dropna().head(10).tolist()
        vals_str = ", ".join(str(v) for v in vals)
        nunique  = int(df[col].nunique())
        n_miss   = int(df[col].isnull().sum())
    else:
        vals_str = "(not in dataset)"
        nunique  = 0
        n_miss   = 0

    tooltip_body = [
        html.Strong(col, style={"fontSize": "0.88rem"}),
        html.Hr(style={"margin": "4px 0", "borderColor": "#4b5563"}),
        html.Div([html.Span("First 10 values: ", style={"color": "#9ca3af"}), vals_str],
                 style={"fontSize": "0.78rem", "wordBreak": "break-all"}),
        html.Div([html.Span("Unique / Missing: ", style={"color": "#9ca3af"}),
                  f"{nunique} / {n_miss}"],
                 style={"fontSize": "0.78rem"}),
    ]

    # ── Inline options row ────────────────────────────────────────────────────
    options_row = None

    if options_str:
        opts = [o.strip() for o in options_str.split(" | ") if o.strip()]
        opt_dropdown_options = [{"label": o, "value": o} for o in opts]
        options_row = html.Div([
            html.Span(
                "Assigned options: ",
                style={"fontSize": "0.68rem", "color": "#9ca3af",
                       "fontWeight": "600", "marginRight": "4px",
                       "whiteSpace": "nowrap"},
            ),
            dcc.Dropdown(
                id={"type": "vm-opt-sel", "index": col},
                options=opt_dropdown_options,
                value=current_opt_assignment or [],
                multi=True,
                placeholder="Select which options this variable represents…",
                style={"fontSize": "0.72rem", "flex": "1"},
            ),
        ], style={"marginTop": "5px", "display": "flex", "alignItems": "center",
                  "gap": "4px"})

    elif scale_str:
        options_row = html.Div(
            [html.Span("Scale: ", style={"fontSize": "0.68rem", "color": "#9ca3af",
                                         "fontWeight": "600", "marginRight": "4px"}),
             html.Span(scale_str, style={"fontSize": "0.72rem", "color": "#1e40af",
                                         "fontStyle": "italic"})],
            style={"marginTop": "5px"},
        )

    # ── Tile styling ──────────────────────────────────────────────────────────
    if is_possibly_related:
        tile_style = {
            "background": "#fffbeb", "border": "1px solid #f59e0b",
            "borderLeft": "3px solid #f59e0b", "borderRadius": "6px",
            "padding": "6px 10px", "marginBottom": "4px",
        }
    elif is_extra:
        tile_style = {
            "background": "#eff6ff", "border": "1px solid #93c5fd",
            "borderLeft": "3px solid #3b82f6", "borderRadius": "6px",
            "padding": "6px 10px", "marginBottom": "4px",
        }
    else:
        tile_style = {
            "background": "#ffffff", "border": "1px solid #e5e7eb",
            "borderLeft": "3px solid #d1d5db", "borderRadius": "6px",
            "padding": "6px 10px", "marginBottom": "4px",
        }

    badge = None
    if is_possibly_related:
        badge = dbc.Badge("possibly related", color="warning", className="ms-1",
                          style={"fontSize": "0.6rem", "verticalAlign": "middle"})
    elif is_extra:
        badge = dbc.Badge("reassigned here", color="primary", className="ms-1",
                          style={"fontSize": "0.6rem", "verticalAlign": "middle"})

    return html.Div([
        dbc.Row([
            # Column name + hover target
            dbc.Col(
                html.Div([
                    html.Span(
                        col,
                        id=tid,
                        style={
                            "fontWeight": "600",
                            "fontSize": "0.82rem",
                            "cursor": "help",
                            "fontFamily": "monospace",
                            "color": "#1e2130",
                        },
                    ),
                    badge,
                ]),
                width=4,
                className="d-flex align-items-center",
            ),
            # Type dropdown
            dbc.Col(
                dcc.Dropdown(
                    id={"type": "vm-type-sel", "index": col},
                    options=VAR_TYPE_OPTIONS,
                    value=current_type,
                    clearable=False,
                    style={"fontSize": "0.76rem", "minWidth": "110px"},
                ),
                width=3,
            ),
            # Move-to dropdown
            dbc.Col(
                dcc.Dropdown(
                    id={"type": "vm-assign-sel", "index": col},
                    options=[{"label": c, "value": c} for c in all_q_codes],
                    value=current_q_code,
                    placeholder="Move to question…",
                    clearable=False,
                    style={"fontSize": "0.76rem"},
                ),
                width=4,
            ),
            # Delete button
            dbc.Col(
                dbc.Button(
                    html.I(className="bi bi-x-lg"),
                    id={"type": "vm-del-btn", "index": col},
                    color="light",
                    size="sm",
                    n_clicks=0,
                    style={"color": "#ef4444", "padding": "2px 8px", "border": "none"},
                    title="Remove this variable from the mapping",
                ),
                width=1,
                className="d-flex justify-content-end align-items-center",
            ),
        ], className="g-1 align-items-center"),
        # Inline options row
        options_row,
        # Tooltip (shown on hover over column name)
        dbc.Tooltip(
            tooltip_body,
            target=tid,
            placement="right",
            style={
                "maxWidth": "340px",
                "textAlign": "left",
                "background": "#1e2130",
                "color": "#e5e7eb",
                "borderRadius": "8px",
                "padding": "10px 12px",
                "fontSize": "0.8rem",
            },
        ),
    ], style=tile_style)


# ── group renderer ─────────────────────────────────────────────────────────────

def _render_groups(groups: list, df: pd.DataFrame, vm_state: dict) -> html.Div:
    deleted            = set(vm_state.get("deleted", []))
    type_overrides     = vm_state.get("type_overrides", {})
    option_assignments = vm_state.get("option_assignments", {})
    all_q_codes        = [g["code"] for g in groups if g["code"] != "_UNMATCHED_"]

    items = []
    for grp in groups:
        code = grp["code"]
        q    = grp["question"]

        active_matched  = [c for c in grp["matched_cols"]     if c not in deleted]
        active_possibly = [c for c in grp["possibly_related"] if c not in deleted]
        active_extra    = [c for c in grp["extra_cols"]        if c not in deleted]

        # Skip QNR-only groups that are also empty (no cols, no reassigned)
        if not active_matched and not active_possibly and not active_extra:
            if grp["status"] == "qnr_only" and code != "_UNMATCHED_":
                # Still show the header so user knows the question exists
                pass

        # ── Card header ────────────────────────────────────────────────────────
        if code == "_UNMATCHED_":
            header_content = html.Div([
                html.Span("UNMATCHED", className="vm-q-code vm-q-code-unmatched"),
                html.Span(
                    "Dataset columns with no questionnaire match",
                    className="vm-q-text",
                    style={"color": "#6b7280"},
                ),
                dbc.Badge(f"{len(active_matched)} col(s)", color="secondary", className="ms-2"),
            ], className="d-flex align-items-center gap-2 flex-wrap")
        else:
            q_type    = q.get("q_type", "") if q else ""
            n_total   = len(active_matched) + len(active_possibly) + len(active_extra)
            sc_color  = "success" if grp["status"] == "matched" else (
                        "warning" if grp["status"] == "no_match" else "secondary")
            sc_label  = (f"{len(active_matched)} col(s)" if active_matched
                         else ("QNR only — no data cols" if grp["status"] == "qnr_only"
                               else "No QNR match"))
            q_text    = (q.get("question", "") if q else "")
            q_display = (q_text[:90] + "…") if len(q_text) > 90 else q_text

            badges = [
                dbc.Badge(q_type, color="info",  className="ms-1") if q_type else None,
                dbc.Badge(grp["scale"], color="light", text_color="dark",
                          className="ms-1", style={"fontSize": "0.63rem"})
                    if grp["scale"] else None,
                dbc.Badge(sc_label, color=sc_color, className="ms-1"),
                dbc.Badge(f"+{len(active_possibly)} possibly related",
                          color="warning", className="ms-1")
                    if active_possibly else None,
            ]

            header_content = html.Div([
                html.Span(code, className="vm-q-code"),
                html.Span(q_display, className="vm-q-text"),
                *[b for b in badges if b],
            ], className="d-flex align-items-center gap-2 flex-wrap")

        # ── Tile body ──────────────────────────────────────────────────────────
        tile_divs = []

        # Build options/scale strings to pass into each tile
        scale_str = grp["scale"]

        for col in active_matched:
            col_type = type_overrides.get(col) or suggest_var_type(
                df, [col], "single", q.get("q_type", "") if q else "")
            tile_divs.append(_render_tile(
                col, df, grp["options_str"], scale_str, col_type, code, all_q_codes,
                False, False, option_assignments.get(col, [])))

        if active_possibly:
            tile_divs.append(html.Div([
                html.I(className="bi bi-exclamation-triangle-fill me-1",
                       style={"color": "#d97706", "fontSize": "0.8rem"}),
                html.Span(
                    "These columns may belong to this question (no exact QNR match found):",
                    style={"fontSize": "0.76rem", "color": "#d97706", "fontWeight": "600"},
                ),
            ], style={"marginTop": "8px", "marginBottom": "4px", "paddingLeft": "2px"}))
            for col in active_possibly:
                col_type = type_overrides.get(col, "categorical")
                tile_divs.append(_render_tile(
                    col, df, grp["options_str"], scale_str, col_type, code, all_q_codes,
                    True, False, option_assignments.get(col, [])))

        for col in active_extra:
            col_type = type_overrides.get(col, "categorical")
            tile_divs.append(_render_tile(
                col, df, grp["options_str"], scale_str, col_type, code, all_q_codes,
                False, True, option_assignments.get(col, [])))

        if not tile_divs:
            tile_divs.append(html.Div(
                "No dataset columns linked to this question.",
                style={"color": "#9ca3af", "fontSize": "0.8rem",
                       "padding": "4px 0", "fontStyle": "italic"},
            ))

        items.append(dbc.AccordionItem(
            children=html.Div(tile_divs, style={"padding": "6px 0"}),
            title=header_content,
            item_id=code,
        ))

    if not items:
        return dbc.Alert("No mapping data to display.", color="secondary")

    return dbc.Accordion(
        items,
        always_open=True,
        start_collapsed=False,
        className="vm-accordion",
        persistence=False,
    )


# ── summary badges ─────────────────────────────────────────────────────────────

def _summary_badges(df: pd.DataFrame, groups: list, vm_state: dict) -> dbc.Row:
    deleted = set(vm_state.get("deleted", []))
    active_total    = sum(1 for c in df.columns if c not in deleted)
    matched_groups  = sum(1 for g in groups if g["status"] == "matched")
    possibly_total  = sum(
        len([c for c in g["possibly_related"] if c not in deleted])
        for g in groups
    )
    qnr_only        = sum(1 for g in groups if g["status"] == "qnr_only"
                          and g["code"] != "_UNMATCHED_")
    unmatched_total = sum(
        len([c for c in g["matched_cols"] if c not in deleted])
        for g in groups if g["code"] == "_UNMATCHED_"
    )

    return dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Active Columns",    className="metric-label"),
            html.Div(active_total,        className="metric-value"),
        ]), className="metric-card"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Questions Matched", className="metric-label"),
            html.Div(matched_groups, className="metric-value", style={"color": "#16a34a"}),
        ]), className="metric-card"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Possibly Related",  className="metric-label"),
            html.Div(possibly_total, className="metric-value", style={"color": "#d97706"}),
        ]), className="metric-card"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("No QNR Match",      className="metric-label"),
            html.Div(unmatched_total, className="metric-value", style={"color": "#6b7280"}),
        ]), className="metric-card"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("QNR Only",          className="metric-label"),
            html.Div(qnr_only, className="metric-value", style={"color": "#6b7280"}),
        ]), className="metric-card"), width=2),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div("Deleted",           className="metric-label"),
            html.Div(len(deleted), className="metric-value", style={"color": "#ef4444"}),
        ]), className="metric-card"), width=2),
    ], className="mb-3")


# ── layout ────────────────────────────────────────────────────────────────────

def layout(state: dict) -> html.Div:
    header = html.Div([
        html.H2("Variable Mapping"),
        html.P(
            "Upload your raw data and questionnaire. "
            "Each question card shows its linked columns as tiles — "
            "hover a column name to preview data, change its type, move it, or delete it.",
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
                    "Drag & drop or ",
                    html.Span("browse",
                              style={"color": "#3a6df0", "textDecoration": "underline",
                                     "cursor": "pointer"}),
                ]),
                accept=".csv,.xlsx,.xls",
                className="upload-area",
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
                    "Drag & drop or ",
                    html.Span("browse",
                              style={"color": "#3a6df0", "textDecoration": "underline",
                                     "cursor": "pointer"}),
                ]),
                accept=".docx,.pdf,.xlsx,.xls,.txt",
                className="upload-area",
            ),
            html.Div(id="vm-qnr-status", className="mt-2"),
        ])), width=6),
    ], className="mb-3")

    return html.Div([
        header,
        upload_row,
        dcc.Store(id="vm-state", data=DEFAULT_VM_STATE, storage_type="memory"),
        html.Div(id="vm-mapping-outer"),
    ])


# ── callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("vm-data-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("vm-upload-data", "contents"),
    State("vm-upload-data", "filename"),
    State("app-state", "data"),
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
        feature_cols = [c for c in df.columns if c != df.columns[-1]]
        auto_num = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        state["numeric_cols"]     = [c for c in feature_cols if c in auto_num]
        state["categorical_cols"] = [c for c in feature_cols if c not in auto_num]
        state["ordinal_cols"]     = []
        if server_store.get_val("qnr_questions"):
            state["mapping_done"] = True
        return (
            dbc.Alert(
                f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from '{filename}'",
                color="success", dismissable=True,
            ),
            state,
        )
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger"), no_update


@callback(
    Output("vm-qnr-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("vm-upload-qnr", "contents"),
    State("vm-upload-qnr", "filename"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def upload_qnr(contents, filename, state):
    if not contents:
        return no_update, no_update
    try:
        ext = pathlib.Path(filename).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        questions = _decode_qnr(contents, filename, tmp.name)
        os.unlink(tmp.name)
        server_store.set_val("qnr_questions", questions)
        state = dict(state or {})
        if server_store.get_df("raw_df") is not None:
            state["mapping_done"] = True
        return (
            dbc.Alert(
                f"Parsed {len(questions)} questions from '{filename}'",
                color="success", dismissable=True,
            ),
            state,
        )
    except Exception as e:
        return dbc.Alert(f"Error: {e}", color="danger"), no_update


@callback(
    Output("vm-mapping-outer", "children"),
    Input("app-state", "data"),
    Input("vm-state", "data"),
)
def refresh_mapping(app_state, vm_state):
    raw_df    = server_store.get_df("raw_df")
    questions = server_store.get_val("qnr_questions", [])
    vm_state  = vm_state or DEFAULT_VM_STATE

    if raw_df is None and not questions:
        return dbc.Alert("Upload both files above to generate the variable mapping.",
                         color="secondary")
    if raw_df is None:
        return dbc.Alert("Questionnaire loaded. Upload the raw data file to see the mapping.",
                         color="info")
    if not questions:
        return dbc.Alert("Raw data loaded. Upload the questionnaire file to see the mapping.",
                         color="info")

    groups = _build_groups(raw_df, questions, vm_state)

    return html.Div([
        _summary_badges(raw_df, groups, vm_state),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5("Variable — Question mapping"), width="auto"),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-arrow-counterclockwise me-1"), "Reset all"],
                        id="vm-reset-btn",
                        color="outline-secondary",
                        size="sm",
                        n_clicks=0,
                        className="me-2",
                    ),
                ], width="auto", className="ms-auto"),
            ], align="center", className="mb-2"),
            html.P(
                "Hover column name to preview data & options  ·  "
                "Change type via dropdown  ·  "
                "Move to another question via 'Move to…'  ·  "
                "✕ to remove from mapping",
                style={"color": "#6b7280", "fontSize": "0.8rem", "marginBottom": "12px"},
            ),
            # Column headers for the tile rows
            dbc.Row([
                dbc.Col(html.Small("Column", style={"color":"#9ca3af","fontWeight":"600"}), width=4),
                dbc.Col(html.Small("Type",   style={"color":"#9ca3af","fontWeight":"600"}), width=3),
                dbc.Col(html.Small("Move to question", style={"color":"#9ca3af","fontWeight":"600"}), width=4),
                dbc.Col(html.Small("Del",   style={"color":"#9ca3af","fontWeight":"600"}), width=1),
            ], className="g-1 mb-1 px-2",
               style={"borderBottom":"1px solid #e5e7eb","paddingBottom":"4px"}),
            _render_groups(groups, raw_df, vm_state),
        ]), className="mb-3"),
    ])


# ── mutation callbacks ────────────────────────────────────────────────────────

@callback(
    Output("vm-state", "data"),
    Input({"type": "vm-del-btn", "index": ALL}, "n_clicks"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_delete(n_clicks_list, vm_state):
    if not ctx.triggered_id or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    col   = ctx.triggered_id["index"]
    state = dict(vm_state or DEFAULT_VM_STATE)
    deleted = list(state.get("deleted", []))
    if col not in deleted:
        deleted.append(col)
    state["deleted"] = deleted
    return state


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Input({"type": "vm-type-sel", "index": ALL}, "value"),
    State({"type": "vm-type-sel", "index": ALL}, "id"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_type_change(values, ids, vm_state):
    if not ctx.triggered_id:
        return no_update
    col = ctx.triggered_id["index"]
    idx = next((i for i, id_ in enumerate(ids) if id_["index"] == col), None)
    if idx is None:
        return no_update
    state    = dict(vm_state or DEFAULT_VM_STATE)
    overrides = dict(state.get("type_overrides", {}))
    overrides[col] = values[idx]
    state["type_overrides"] = overrides
    return state


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Input({"type": "vm-assign-sel", "index": ALL}, "value"),
    State({"type": "vm-assign-sel", "index": ALL}, "id"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_reassign(values, ids, vm_state):
    if not ctx.triggered_id:
        return no_update
    col = ctx.triggered_id["index"]
    idx = next((i for i, id_ in enumerate(ids) if id_["index"] == col), None)
    if idx is None or values[idx] is None:
        return no_update
    state     = dict(vm_state or DEFAULT_VM_STATE)
    reassigned = dict(state.get("reassigned", {}))
    reassigned[col] = values[idx]
    state["reassigned"] = reassigned
    return state


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Input({"type": "vm-opt-sel", "index": ALL}, "value"),
    State({"type": "vm-opt-sel", "index": ALL}, "id"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_option_assign(values, ids, vm_state):
    if not ctx.triggered_id:
        return no_update
    col = ctx.triggered_id["index"]
    idx = next((i for i, id_ in enumerate(ids) if id_["index"] == col), None)
    if idx is None:
        return no_update
    state = dict(vm_state or DEFAULT_VM_STATE)
    assignments = dict(state.get("option_assignments", {}))
    assignments[col] = values[idx] or []
    state["option_assignments"] = assignments
    return state


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Input("vm-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_vm_state(n):
    if n:
        return DEFAULT_VM_STATE
    return no_update
