"""
Variable Mapping — pre-Stage 0
One accordion card per question; each card body is a table where every
row is a dataset column.  Questionnaire options appear as draggable chips
in column 2 — drop them onto any row to reassign.  Type and question-
mapping dropdowns sit in columns 3–4.  Click "Save changes" to commit.
"""
import base64, io, re, tempfile, pathlib
import pandas as pd
import numpy as np

from dash import (dcc, html, Input, Output, State,
                  callback, clientside_callback, no_update, ctx, ALL)
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

DEFAULT_VM_STATE = {
    "deleted": [], "type_overrides": {}, "reassigned": {}, "option_assignments": {}
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_id(col: str) -> str:
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


def _tooltip_content(col: str, df: pd.DataFrame) -> list:
    if col in df.columns:
        vals     = df[col].dropna().head(10).tolist()
        vals_str = ", ".join(str(v) for v in vals)
        nunique  = int(df[col].nunique())
        n_miss   = int(df[col].isnull().sum())
    else:
        vals_str = "(not in dataset)"
        nunique  = 0
        n_miss   = 0
    return [
        html.Strong(col, style={"fontSize": "0.88rem"}),
        html.Hr(style={"margin": "4px 0", "borderColor": "#4b5563"}),
        html.Div([html.Span("First 10 values: ", style={"color": "#9ca3af"}), vals_str],
                 style={"fontSize": "0.78rem", "wordBreak": "break-all"}),
        html.Div([html.Span("Unique / Missing: ", style={"color": "#9ca3af"}),
                  f"{nunique} / {n_miss}"],
                 style={"fontSize": "0.78rem"}),
    ]


# ── default option assignment ─────────────────────────────────────────────────

def _default_option_assignments(df: pd.DataFrame, groups: list) -> dict:
    """
    Auto-assign questionnaire options to dataset columns using structural cues.

    Rules (priority order):
      scale_* / numeric  → skip  (scale label shown, not chips)
      1 col per question → all options  (column stores the answer code)
      multi q_type, N cols → numeric suffix _1/_2 or letter a/b → opts[i]
      grid, N cols       → all options per col (same scale, different row item)
      other multi-col    → all options per col  (safe fallback)
    """
    defaults: dict = {}

    for grp in groups:
        q = grp["question"]
        if not q:
            continue
        opts   = q.get("options", [])
        q_type = q.get("q_type", "single")

        if q_type.startswith("scale_") or q_type == "numeric":
            continue
        if not opts:
            continue

        all_cols = grp["matched_cols"] + grp["possibly_related"] + grp["extra_cols"]
        if not all_cols:
            continue

        if len(all_cols) == 1:
            defaults[all_cols[0]] = list(opts)

        elif q_type == "multi":
            for col in all_cols:
                # numeric suffix: A3_1 → idx 0,  A3_2 → idx 1
                m_num = re.search(r'[_\-](\d+)$', col)
                # letter suffix: A3a → idx 0,  A3b → idx 1
                m_let = re.search(r'[A-Z]\d+([a-z])$', col, re.IGNORECASE)
                if m_num:
                    idx = int(m_num.group(1)) - 1
                    if 0 <= idx < len(opts):
                        defaults[col] = [opts[idx]]
                        continue
                if m_let:
                    idx = ord(m_let.group(1).lower()) - ord('a')
                    if 0 <= idx < len(opts):
                        defaults[col] = [opts[idx]]
                        continue
                defaults[col] = list(opts)

        else:
            # grid or single with multiple cols → all options per col
            for col in all_cols:
                defaults[col] = list(opts)

    return defaults


# ── data grouping ─────────────────────────────────────────────────────────────

def _build_groups(df: pd.DataFrame, questions: list, vm_state: dict) -> list:
    reassigned = vm_state.get("reassigned", {})
    q_map      = {q["code"].upper(): q for q in questions}
    col_groups = group_columns(df)

    used_cols: set = set()
    groups = []

    for q in questions:
        code = q["code"].upper()
        grp  = col_groups.get(code, {})
        orig = grp.get("cols", [])
        used_cols.update(orig)

        matched_cols = [c for c in orig if reassigned.get(c, code) == code]

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

        extra_cols = [
            c for c, tgt in reassigned.items()
            if tgt == code and c not in matched_cols and c not in possibly_related
        ]

        scale = (
            f"{q['scale_points']}-pt "
            f"[{q.get('scale_low', '')} — {q.get('scale_high', '')}]"
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


# ── table renderer ────────────────────────────────────────────────────────────

def _render_var_table(
    grp: dict,
    df: pd.DataFrame,
    vm_state: dict,
    default_assignments: dict,
    all_q_codes: list,
) -> html.Div:
    """One table per question group — each row is a dataset column."""
    deleted        = set(vm_state.get("deleted", []))
    type_overrides = vm_state.get("type_overrides", {})
    user_assigns   = vm_state.get("option_assignments", {})

    code = grp["code"]
    q    = grp["question"]

    def opt_for(col):
        # Explicit user save wins; else auto-computed default
        return user_assigns[col] if col in user_assigns else default_assignments.get(col, [])

    # Collect active columns with their role flag
    col_roles: list[tuple] = []   # (col, is_possibly, is_extra)
    for col in grp["matched_cols"]:
        if col not in deleted:
            col_roles.append((col, False, False))
    for col in grp["possibly_related"]:
        if col not in deleted:
            col_roles.append((col, True, False))
    for col in grp["extra_cols"]:
        if col not in deleted:
            col_roles.append((col, False, True))

    if not col_roles:
        return html.Div(
            "No dataset columns linked to this question.",
            style={"color": "#9ca3af", "fontSize": "0.8rem",
                   "fontStyle": "italic", "padding": "4px 0"},
        )

    # Unassigned options pool (options not yet placed on any row)
    q_opts       = q.get("options", []) if q else []
    assigned_set = {o for col, _, _ in col_roles for o in opt_for(col)}
    pool         = [o for o in q_opts if o not in assigned_set]

    TD = {"padding": "5px 8px", "verticalAlign": "middle"}

    rows = []
    for col, is_possibly, is_extra in col_roles:
        col_type = type_overrides.get(col) or suggest_var_type(
            df, [col], "single", q.get("q_type", "") if q else "")

        chips = [
            html.Span(
                opt,
                className="opt-chip",
                draggable="true",
                **{"data-opt": opt},
            )
            for opt in opt_for(col)
        ]

        if is_possibly:
            badge  = dbc.Badge("~related", color="warning", className="ms-1",
                               style={"fontSize": "0.58rem"})
            row_bg = "#fffbeb"
        elif is_extra:
            badge  = dbc.Badge("reassigned", color="primary", className="ms-1",
                               style={"fontSize": "0.58rem"})
            row_bg = "#eff6ff"
        else:
            badge  = None
            row_bg = None

        td = {**TD, **({"background": row_bg} if row_bg else {})}

        rows.append(html.Tr([
            # C1 — variable name + tooltip + delete
            html.Td(
                html.Div([
                    html.Span(
                        col,
                        id=_safe_id(col),
                        style={"fontFamily": "monospace", "fontWeight": "600",
                               "fontSize": "0.81rem", "cursor": "help",
                               "color": "#1e2130"},
                    ),
                    badge,
                    dbc.Button(
                        html.I(className="bi bi-x-lg"),
                        id={"type": "vm-del-btn", "index": col},
                        color="link", size="sm", n_clicks=0,
                        style={"color": "#ef4444", "padding": "0 0 0 6px",
                               "lineHeight": "1", "border": "none",
                               "marginLeft": "auto"},
                        title="Remove from mapping",
                    ),
                    dbc.Tooltip(
                        _tooltip_content(col, df),
                        target=_safe_id(col),
                        placement="right",
                        style={"maxWidth": "320px", "background": "#1e2130",
                               "color": "#e5e7eb", "borderRadius": "8px",
                               "padding": "10px 12px", "fontSize": "0.78rem"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "2px"}),
                style={**td, "width": "180px", "whiteSpace": "nowrap"},
            ),

            # C2 — draggable option chips (drop zone)
            html.Td(
                html.Div(
                    chips or [],
                    className="opt-dropzone",
                    **{"data-col": col},
                ),
                style={**td, "minWidth": "220px"},
            ),

            # C3 — variable type
            html.Td(
                dcc.Dropdown(
                    id={"type": "vm-type-sel", "index": col},
                    options=VAR_TYPE_OPTIONS,
                    value=col_type,
                    clearable=False,
                    style={"fontSize": "0.76rem", "minWidth": "110px"},
                ),
                style={**td, "width": "135px"},
            ),

            # C4 — move to question
            html.Td(
                dcc.Dropdown(
                    id={"type": "vm-assign-sel", "index": col},
                    options=[{"label": c, "value": c} for c in all_q_codes],
                    value=code if code != "_UNMATCHED_" else None,
                    placeholder="Move to…",
                    clearable=False,
                    style={"fontSize": "0.76rem", "minWidth": "120px"},
                ),
                style={**td, "width": "155px"},
            ),
        ]))

    # Optional unassigned-options pool row
    if pool:
        rows.append(html.Tr([
            html.Td(
                html.Span("unassigned",
                          style={"color": "#9ca3af", "fontSize": "0.74rem",
                                 "fontStyle": "italic"}),
                style={"padding": "5px 8px", "background": "#f9fafb",
                       "verticalAlign": "middle", "width": "180px"},
            ),
            html.Td(
                html.Div(
                    [html.Span(o, className="opt-chip opt-chip-pool",
                               draggable="true", **{"data-opt": o})
                     for o in pool],
                    className="opt-dropzone",
                    **{"data-col": "__pool__"},
                ),
                style={"padding": "5px 8px", "background": "#f9fafb"},
            ),
            html.Td(style={"background": "#f9fafb"}),
            html.Td(style={"background": "#f9fafb"}),
        ]))

    return html.Table([
        html.Thead(
            html.Tr([
                html.Th("Variable",
                        style={"fontSize": "0.71rem", "color": "#9ca3af",
                               "fontWeight": "600", "padding": "4px 8px",
                               "width": "180px"}),
                html.Th("Options — drag to reassign",
                        style={"fontSize": "0.71rem", "color": "#9ca3af",
                               "fontWeight": "600", "padding": "4px 8px"}),
                html.Th("Type",
                        style={"fontSize": "0.71rem", "color": "#9ca3af",
                               "fontWeight": "600", "padding": "4px 8px",
                               "width": "135px"}),
                html.Th("Question",
                        style={"fontSize": "0.71rem", "color": "#9ca3af",
                               "fontWeight": "600", "padding": "4px 8px",
                               "width": "155px"}),
            ]),
            style={"borderBottom": "1px solid #e5e7eb"},
        ),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "marginTop": "6px", "tableLayout": "fixed"})


# ── group renderer (accordion) ────────────────────────────────────────────────

def _render_groups(groups: list, df: pd.DataFrame, vm_state: dict,
                   default_assignments: dict = None) -> html.Div:
    deleted             = set(vm_state.get("deleted", []))
    default_assignments = default_assignments or {}
    all_q_codes         = [g["code"] for g in groups if g["code"] != "_UNMATCHED_"]

    items = []
    for grp in groups:
        code = grp["code"]
        q    = grp["question"]

        active_matched  = [c for c in grp["matched_cols"]     if c not in deleted]
        active_possibly = [c for c in grp["possibly_related"] if c not in deleted]
        active_extra    = [c for c in grp["extra_cols"]        if c not in deleted]

        # ── Accordion header ───────────────────────────────────────────────────
        if code == "_UNMATCHED_":
            header_content = html.Div([
                html.Span("UNMATCHED", className="vm-q-code vm-q-code-unmatched"),
                html.Span("Dataset columns with no questionnaire match",
                          className="vm-q-text", style={"color": "#6b7280"}),
                dbc.Badge(f"{len(active_matched)} col(s)",
                          color="secondary", className="ms-2"),
            ], className="d-flex align-items-center gap-2 flex-wrap")
        else:
            q_type    = q.get("q_type", "") if q else ""
            sc_color  = ("success" if grp["status"] == "matched" else
                         "warning" if grp["status"] == "no_match" else "secondary")
            sc_label  = (f"{len(active_matched)} col(s)" if active_matched else
                         "QNR only — no data cols" if grp["status"] == "qnr_only"
                         else "No QNR match")
            q_text    = q.get("question", "") if q else ""
            q_display = (q_text[:90] + "…") if len(q_text) > 90 else q_text

            badges = [
                dbc.Badge(q_type, color="info", className="ms-1") if q_type else None,
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

        items.append(dbc.AccordionItem(
            children=_render_var_table(grp, df, vm_state, default_assignments, all_q_codes),
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
    deleted         = set(vm_state.get("deleted", []))
    active_total    = sum(1 for c in df.columns if c not in deleted)
    matched_groups  = sum(1 for g in groups if g["status"] == "matched")
    possibly_total  = sum(
        len([c for c in g["possibly_related"] if c not in deleted]) for g in groups)
    qnr_only        = sum(1 for g in groups
                          if g["status"] == "qnr_only" and g["code"] != "_UNMATCHED_")
    unmatched_total = sum(
        len([c for c in g["matched_cols"] if c not in deleted])
        for g in groups if g["code"] == "_UNMATCHED_")

    def _card(label, value, color=None):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value",
                     style=({"color": color} if color else {})),
        ]), className="metric-card"), width=2)

    return dbc.Row([
        _card("Active Columns",    active_total),
        _card("Questions Matched", matched_groups,  "#16a34a"),
        _card("Possibly Related",  possibly_total,  "#d97706"),
        _card("No QNR Match",      unmatched_total, "#6b7280"),
        _card("QNR Only",          qnr_only,        "#6b7280"),
        _card("Deleted",           len(deleted),    "#ef4444"),
    ], className="mb-3")


# ── layout ────────────────────────────────────────────────────────────────────

def layout(state: dict) -> html.Div:
    header = html.Div([
        html.H2("Variable Mapping"),
        html.P(
            "Upload your raw data and questionnaire. "
            "Each question card shows a table of its linked columns — "
            "drag option chips between rows to reassign them to a different variable, "
            "then click Save changes.",
            style={"color": "#6b7280"},
        ),
    ], className="stage-header")

    upload_row = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6([html.I(className="bi bi-table me-2"),
                     "Raw data (CSV / Excel)"], className="card-title"),
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
        dcc.Store(id="vm-state", data=DEFAULT_VM_STATE, storage_type="session"),
        dcc.Store(id="vm-drag-assignments", data=None),
        html.Div(id="vm-save-feedback", className="mb-1"),
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

    groups      = _build_groups(raw_df, questions, vm_state)
    def_assigns = _default_option_assignments(raw_df, groups)

    return html.Div([
        _summary_badges(raw_df, groups, vm_state),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H5("Variable — Question mapping"), width="auto"),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-floppy me-1"), "Save changes"],
                        id="vm-save-btn",
                        color="primary",
                        size="sm",
                        n_clicks=0,
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-counterclockwise me-1"), "Reset all"],
                        id="vm-reset-btn",
                        color="outline-secondary",
                        size="sm",
                        n_clicks=0,
                    ),
                ], width="auto", className="ms-auto"),
            ], align="center", className="mb-2"),
            html.P(
                "Hover a column name to preview its data  ·  "
                "Drag option chips between rows to reassign them  ·  "
                "Click Save changes to commit.",
                style={"color": "#6b7280", "fontSize": "0.8rem", "marginBottom": "10px"},
            ),
            _render_groups(groups, raw_df, vm_state, def_assigns),
        ]), className="mb-3"),
    ])


# ── clientside: read drag state from DOM on Save ──────────────────────────────

clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var assignments = {};
        document.querySelectorAll('.opt-dropzone').forEach(function(zone) {
            var col = zone.getAttribute('data-col');
            if (!col || col === '__pool__') return;
            var opts = [];
            zone.querySelectorAll('.opt-chip').forEach(function(chip) {
                opts.push(chip.getAttribute('data-opt'));
            });
            assignments[col] = opts;
        });
        return assignments;
    }
    """,
    Output("vm-drag-assignments", "data"),
    Input("vm-save-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ── server callbacks ──────────────────────────────────────────────────────────

@callback(
    Output("vm-state", "data"),
    Input({"type": "vm-del-btn", "index": ALL}, "n_clicks"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_delete(n_clicks_list, vm_state):
    if not ctx.triggered_id or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    col     = ctx.triggered_id["index"]
    state   = dict(vm_state or DEFAULT_VM_STATE)
    deleted = list(state.get("deleted", []))
    if col not in deleted:
        deleted.append(col)
    state["deleted"] = deleted
    return state


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Output("vm-save-feedback", "children"),
    Input("vm-drag-assignments", "data"),
    State({"type": "vm-type-sel",   "index": ALL}, "value"),
    State({"type": "vm-type-sel",   "index": ALL}, "id"),
    State({"type": "vm-assign-sel", "index": ALL}, "value"),
    State({"type": "vm-assign-sel", "index": ALL}, "id"),
    State("vm-state", "data"),
    prevent_initial_call=True,
)
def handle_save(drag_assigns, type_vals, type_ids,
                assign_vals, assign_ids, vm_state):
    if drag_assigns is None:
        return no_update, no_update
    state = dict(vm_state or DEFAULT_VM_STATE)

    # Option assignments from drag-and-drop DOM snapshot.
    # Only overwrite if the clientside callback actually found dropzones;
    # an empty dict means no dropzones were in the DOM — preserve existing.
    if drag_assigns:
        state["option_assignments"] = {
            col: opts for col, opts in drag_assigns.items() if opts is not None
        }

    # Type overrides
    overrides = dict(state.get("type_overrides", {}))
    for val, id_ in zip(type_vals, type_ids):
        if val:
            overrides[id_["index"]] = val
    state["type_overrides"] = overrides

    # Question reassignments
    reassigned = dict(state.get("reassigned", {}))
    for val, id_ in zip(assign_vals, assign_ids):
        if val:
            reassigned[id_["index"]] = val
    state["reassigned"] = reassigned

    return state, dbc.Alert(
        [html.I(className="bi bi-check-circle me-2"), "Changes saved."],
        color="success", dismissable=True, duration=3000,
        style={"padding": "6px 12px", "fontSize": "0.82rem"},
    )


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Input("vm-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_vm_state(n):
    if n:
        return DEFAULT_VM_STATE
    return no_update
