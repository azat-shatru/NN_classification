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
from utils.col_mapper import (group_columns, suggest_var_type,
                               parse_excel_metadata, build_questions_from_metadata)

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
    "deleted": [], "type_overrides": {}, "reassigned": {}, "option_assignments": {},
    "deleted_questions": [], "deleted_options": {},
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
                # No suffix match — leave unassigned (user assigns manually)

        else:
            # grid or other multi-col: assign 1 option per column in order.
            # Extra options that exceed number of columns stay in the pool.
            for i, col in enumerate(all_cols):
                if i < len(opts):
                    defaults[col] = [opts[i]]

    # ── Grid-prefix expansion: expand options × table column headers ─────────
    for grp in groups:
        q = grp["question"]
        if not q:
            continue
        if q.get("table_n_cols", 1) <= 2 or not q.get("table_col_headers"):
            continue

        n_opts = len(q.get("options", []))
        n_extra_cols = q["table_n_cols"] - 1   # columns 2+
        col_headers = q["table_col_headers"]    # list of n_extra_cols strings
        all_cols = grp["matched_cols"] + grp["extra_cols"]  # NOT possibly_related
        n_vars = len(all_cols)

        if n_vars > 0 and n_opts > 0 and abs(n_vars - n_opts * n_extra_cols) <= max(1, round(n_opts * n_extra_cols * 0.1)):
            # Expand: for each opt, for each col_header → "col_header opt"
            expanded_opts = []
            for opt in q["options"]:
                for ch in col_headers:
                    expanded_opts.append(f"{ch} {opt}")

            if len(expanded_opts) == n_vars:
                for i, col in enumerate(all_cols):
                    defaults[col] = [expanded_opts[i]]
            else:
                # Assign as many as fit
                for i, col in enumerate(all_cols):
                    if i < len(expanded_opts):
                        defaults[col] = [expanded_opts[i]]

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
    del_opts       = set(vm_state.get("deleted_options", {}).get(grp["code"], []))

    code = grp["code"]
    q    = grp["question"]

    def opt_for(col):
        # Explicit user save wins; else auto-computed default
        raw = user_assigns[col] if col in user_assigns else default_assignments.get(col, [])
        return [o for o in raw if o not in del_opts]

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
    q_opts       = [o for o in (q.get("options", []) if q else []) if o not in del_opts]
    assigned_set = {o for col, _, _ in col_roles for o in opt_for(col)}
    pool         = [o for o in q_opts if o not in assigned_set]

    TD = {"padding": "5px 8px", "verticalAlign": "middle"}

    def _chip(opt, extra_cls=""):
        return html.Span(
            [
                opt,
                html.Button(
                    "×",
                    id={"type": "vm-del-opt-btn", "index": f"{code}||{opt}"},
                    n_clicks=0,
                    className=f"opt-chip-del{' ' + extra_cls if extra_cls else ''}",
                ),
            ],
            className=f"opt-chip{' opt-chip-pool' if extra_cls else ''}",
            draggable="true",
            **{"data-opt": opt},
        )

    # ── Pool row (at top so unassigned options are immediately visible) ──────────
    safe_code = re.sub(r"[^A-Za-z0-9]", "-", code)
    pool_row = html.Tr([
        html.Td(
            html.Div([
                html.Span("unassigned",
                          style={"color": "#9ca3af", "fontSize": "0.74rem",
                                 "fontStyle": "italic", "marginRight": "6px"}),
                # "Add options" toggle button
                html.Button(
                    "+ Add",
                    className="vm-add-opts-btn",
                    **{"data-qcode": safe_code},
                    title="Add custom options by typing or pasting",
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            style={"padding": "5px 8px", "background": "#f9fafb",
                   "verticalAlign": "middle", "width": "180px"},
        ),
        html.Td(
            html.Div(
                [_chip(o, extra_cls="pool") for o in pool],
                className="opt-dropzone",
                **{"data-col": "__pool__"},
            ),
            style={"padding": "5px 8px", "background": "#f9fafb"},
        ),
        html.Td(style={"background": "#f9fafb"}),
        html.Td(style={"background": "#f9fafb"}),
    ])

    # Paste/type panel (hidden by default; toggled by JS)
    paste_row = html.Tr([
        html.Td(
            html.Span("Paste or type options (one per line):",
                      style={"fontSize": "0.72rem", "color": "#6b7280"}),
            style={"padding": "4px 8px", "background": "#f0f4ff",
                   "verticalAlign": "top"},
        ),
        html.Td(
            html.Div([
                html.Textarea(
                    placeholder="Paste here…\nOne option per line",
                    className="vm-opts-textarea",
                    **{"data-qcode": safe_code},
                    rows=4,
                ),
                html.Button(
                    "Add to pool",
                    className="vm-opts-add-btn",
                    **{"data-qcode": safe_code},
                ),
            ]),
            colSpan=3,
            style={"padding": "4px 8px", "background": "#f0f4ff"},
        ),
    ], className="vm-paste-row", style={"display": "none"},
       **{"data-qcode": safe_code})

    rows = []
    for col, is_possibly, is_extra in col_roles:
        col_type = type_overrides.get(col) or suggest_var_type(
            df, [col], "single", q.get("q_type", "") if q else "")

        chips = [_chip(opt) for opt in opt_for(col)]

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
        ], id="vm-row-" + _safe_id(col)))

    return html.Table([
        html.Thead(
            html.Tr([
                html.Th("Variable",
                        style={"fontSize": "0.71rem", "color": "#9ca3af",
                               "fontWeight": "600", "padding": "4px 8px",
                               "width": "180px"}),
                html.Th("Options — drag to reassign  ·  dbl-click chip to edit",
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
        html.Tbody([pool_row, paste_row] + rows),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "marginTop": "6px", "tableLayout": "fixed"})


# ── group renderer (accordion) ────────────────────────────────────────────────

def _render_groups(groups: list, df: pd.DataFrame, vm_state: dict,
                   default_assignments: dict = None) -> html.Div:
    deleted             = set(vm_state.get("deleted", []))
    deleted_questions   = set(vm_state.get("deleted_questions", []))
    default_assignments = default_assignments or {}
    all_q_codes         = [g["code"] for g in groups if g["code"] != "_UNMATCHED_"]

    items = []
    for grp in groups:
        code = grp["code"]
        q    = grp["question"]

        # Skip questions that have been deleted and saved
        if code in deleted_questions:
            continue

        active_matched  = [c for c in grp["matched_cols"]     if c not in deleted]
        active_possibly = [c for c in grp["possibly_related"] if c not in deleted]
        active_extra    = [c for c in grp["extra_cols"]        if c not in deleted]

        # ── Delete-question button (not shown for _UNMATCHED_) ─────────────────
        del_q_btn = (
            html.Button(
                "✕",
                id={"type": "vm-del-q-btn", "index": code},
                n_clicks=0,
                className="vm-del-q-btn",
                title="Remove this question and its variables",
            ) if code != "_UNMATCHED_" else None
        )

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
            # Show first 70 chars in header; full text shown at top of body
            q_preview = (q_text[:70] + "…") if len(q_text) > 70 else q_text

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
                html.Span(q_preview, className="vm-q-text"),
                *[b for b in badges if b],
                del_q_btn,
            ], className="d-flex align-items-center gap-2 flex-wrap")

            # Full question text block shown at top of accordion body
            q_text_block = html.P(
                q_text,
                className="vm-q-fulltext",
            ) if q_text else None

        body_content = html.Div([
            q_text_block,
            _render_var_table(grp, df, vm_state, default_assignments, all_q_codes),
        ]) if (code != "_UNMATCHED_" and q_text_block) else \
            _render_var_table(grp, df, vm_state, default_assignments, all_q_codes)

        items.append(dbc.AccordionItem(
            children=body_content,
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
        dcc.Store(id="vm-pending-deletes", data=[], storage_type="memory"),
        dcc.Store(id="vm-pending-delete-questions", data=[], storage_type="memory"),
        dcc.Store(id="vm-pending-delete-opts", data={}, storage_type="memory"),
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
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8", errors="replace")))
        else:
            df = pd.read_excel(io.BytesIO(decoded))

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

        # ── Auto-build questions from Excel metadata sheets ───────────────────
        meta_msg = ""
        if filename.lower().endswith((".xlsx", ".xls")):
            import io as _io
            meta = parse_excel_metadata(_io.BytesIO(decoded))
            if meta["has_metadata"]:
                col_groups = group_columns(df)
                questions  = build_questions_from_metadata(
                    meta["var_labels"], meta["value_labels"], col_groups
                )
                server_store.set_val("qnr_questions", questions)
                server_store.set_val("excel_metadata", meta)
                state["mapping_done"] = True
                meta_msg = (f"  ·  Auto-mapped {len(questions)} questions "
                            f"from Excel metadata sheets.")

        if not meta_msg and server_store.get_val("qnr_questions"):
            state["mapping_done"] = True

        return (
            dbc.Alert(
                f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns "
                f"from '{filename}'{meta_msg}",
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
            // Exclude chips that are queued for deletion
            zone.querySelectorAll('.opt-chip:not(.opt-chip--deleted)').forEach(function(chip) {
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


# ── clientside: toggle variable-row deletion (grey ↔ restore) ────────────────

clientside_callback(
    """
    function(n_clicks_list, pending) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var hasClick = (n_clicks_list || []).some(function(n) { return n > 0; });
        if (!hasClick) return window.dash_clientside.no_update;

        var prop = ctx.triggered[0].prop_id;
        var col;
        try {
            col = JSON.parse(prop.split('.n_clicks')[0]).index;
        } catch(e) { return window.dash_clientside.no_update; }

        var list = Array.isArray(pending) ? pending.slice() : [];
        var idx  = list.indexOf(col);
        var safeId = 'vm-row-vmt-' + col.replace(/[^A-Za-z0-9]/g, '-');
        var row  = document.getElementById(safeId);

        if (idx !== -1) {
            // Already queued → revert
            list.splice(idx, 1);
            if (row) { row.style.opacity = ''; row.style.textDecoration = ''; }
        } else {
            // Queue for deletion
            list.push(col);
            if (row) { row.style.opacity = '0.35'; row.style.textDecoration = 'line-through'; }
        }
        return list;
    }
    """,
    Output("vm-pending-deletes", "data"),
    Input({"type": "vm-del-btn", "index": ALL}, "n_clicks"),
    State("vm-pending-deletes", "data"),
    prevent_initial_call=True,
)


# ── clientside: toggle question-card deletion (grey ↔ restore) ───────────────

clientside_callback(
    """
    function(n_clicks_list, pending) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var hasClick = (n_clicks_list || []).some(function(n) { return n > 0; });
        if (!hasClick) return window.dash_clientside.no_update;

        var prop = ctx.triggered[0].prop_id;
        var code;
        try {
            code = JSON.parse(prop.split('.n_clicks')[0]).index;
        } catch(e) { return window.dash_clientside.no_update; }

        var list = Array.isArray(pending) ? pending.slice() : [];
        var idx  = list.indexOf(code);

        // Find the accordion-item containing the .vm-q-code span with this text
        var targetItem = null;
        document.querySelectorAll('.accordion-item').forEach(function(item) {
            item.querySelectorAll('.vm-q-code').forEach(function(span) {
                if (span.textContent.trim() === code) targetItem = item;
            });
        });

        if (idx !== -1) {
            // Already queued → revert (remove greying, restore interactivity)
            list.splice(idx, 1);
            if (targetItem) {
                targetItem.style.opacity = '';
                targetItem.style.transition = '';
            }
        } else {
            // Queue for deletion (grey out but keep pointer events so user can toggle back)
            list.push(code);
            if (targetItem) {
                targetItem.style.opacity = '0.3';
                targetItem.style.transition = 'opacity 0.2s';
            }
        }
        return list;
    }
    """,
    Output("vm-pending-delete-questions", "data"),
    Input({"type": "vm-del-q-btn", "index": ALL}, "n_clicks"),
    State("vm-pending-delete-questions", "data"),
    prevent_initial_call=True,
)


# ── clientside: toggle option-chip deletion (.opt-chip--deleted ↔ normal) ────

clientside_callback(
    """
    function(n_clicks_list, pending) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) return window.dash_clientside.no_update;
        var hasClick = (n_clicks_list || []).some(function(n) { return n > 0; });
        if (!hasClick) return window.dash_clientside.no_update;

        var prop = ctx.triggered[0].prop_id;
        var index;
        try {
            index = JSON.parse(prop.split('.n_clicks')[0]).index;
        } catch(e) { return window.dash_clientside.no_update; }

        // index is "QCODE||opt_text"
        var sep = index.indexOf('||');
        if (sep === -1) return window.dash_clientside.no_update;
        var qcode = index.substring(0, sep);
        var opt   = index.substring(sep + 2);

        // Toggle .opt-chip--deleted on the chip
        var btnId = '{"index":"' + index + '","type":"vm-del-opt-btn"}';
        var btn   = document.getElementById(btnId);
        if (btn) {
            var chip = btn.closest('.opt-chip');
            if (chip) chip.classList.toggle('opt-chip--deleted');
        }

        // Toggle in pending map
        var map = (pending && typeof pending === 'object' && !Array.isArray(pending))
                  ? Object.assign({}, pending) : {};
        var optList = Array.isArray(map[qcode]) ? map[qcode].slice() : [];
        var optIdx  = optList.indexOf(opt);
        if (optIdx !== -1) {
            optList.splice(optIdx, 1);   // revert
        } else {
            optList.push(opt);           // queue
        }
        if (optList.length > 0) {
            map[qcode] = optList;
        } else {
            delete map[qcode];
        }
        return map;
    }
    """,
    Output("vm-pending-delete-opts", "data"),
    Input({"type": "vm-del-opt-btn", "index": ALL}, "n_clicks"),
    State("vm-pending-delete-opts", "data"),
    prevent_initial_call=True,
)


# ── server callbacks ──────────────────────────────────────────────────────────

@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Output("vm-save-feedback", "children"),
    Output("vm-pending-deletes", "data", allow_duplicate=True),
    Output("vm-pending-delete-questions", "data", allow_duplicate=True),
    Output("vm-pending-delete-opts", "data", allow_duplicate=True),
    Input("vm-drag-assignments", "data"),
    State({"type": "vm-type-sel",   "index": ALL}, "value"),
    State({"type": "vm-type-sel",   "index": ALL}, "id"),
    State({"type": "vm-assign-sel", "index": ALL}, "value"),
    State({"type": "vm-assign-sel", "index": ALL}, "id"),
    State("vm-state", "data"),
    State("vm-pending-deletes", "data"),
    State("vm-pending-delete-questions", "data"),
    State("vm-pending-delete-opts", "data"),
    prevent_initial_call=True,
)
def handle_save(drag_assigns, type_vals, type_ids,
                assign_vals, assign_ids, vm_state,
                pending_deletes, pending_del_questions, pending_del_opts):
    if drag_assigns is None:
        return no_update, no_update, no_update, no_update, no_update
    state = dict(vm_state or DEFAULT_VM_STATE)

    # Merge pending variable deletes
    deleted = list(state.get("deleted", []))
    for col in (pending_deletes or []):
        if col not in deleted:
            deleted.append(col)

    # Merge pending question deletes — also mark all their columns as deleted
    del_questions = list(state.get("deleted_questions", []))
    raw_df    = server_store.get_df("raw_df")
    questions = server_store.get_val("qnr_questions", [])
    for qcode in (pending_del_questions or []):
        if qcode not in del_questions:
            del_questions.append(qcode)
        # Auto-delete all columns belonging to this question
        if raw_df is not None and questions:
            grps = _build_groups(raw_df, questions, state)
            for grp in grps:
                if grp["code"] == qcode:
                    all_cols = (grp["matched_cols"] + grp["possibly_related"]
                                + grp["extra_cols"])
                    for col in all_cols:
                        if col not in deleted:
                            deleted.append(col)
    state["deleted"]           = deleted
    state["deleted_questions"] = del_questions

    # Merge pending option deletes: {qcode: [opt, ...]}
    del_opts = dict(state.get("deleted_options", {}))
    for qcode, opts in (pending_del_opts or {}).items():
        existing = list(del_opts.get(qcode, []))
        for o in opts:
            if o not in existing:
                existing.append(o)
        del_opts[qcode] = existing
    state["deleted_options"] = del_opts

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
    ), [], [], {}


@callback(
    Output("vm-state", "data", allow_duplicate=True),
    Output("vm-pending-deletes", "data", allow_duplicate=True),
    Output("vm-pending-delete-questions", "data", allow_duplicate=True),
    Output("vm-pending-delete-opts", "data", allow_duplicate=True),
    Input("vm-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_vm_state(n):
    if n:
        return DEFAULT_VM_STATE, [], [], {}
    return no_update, no_update, no_update, no_update
