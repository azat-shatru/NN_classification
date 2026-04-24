"""
Stage 2.6 — Charts Portal
Charts from raw data using QNR-mapped option labels.
Reads codebook from Variable Mapping; auto-builds if absent.
"""
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from dash import dcc, html, Input, Output, State, callback, no_update, ALL, ctx, dash_table
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store
from utils.col_mapper import build_codebook
from utils.ppt_export import build_pptx


# ── Chart menu per variable type ──────────────────────────────────────────────

CHART_OPTIONS = {
    "categorical": ["100% Stacked bar", "Bar chart", "Horizontal bar", "Pie chart", "Donut chart"],
    "single":      ["100% Stacked bar", "Bar chart", "Horizontal bar", "Pie chart", "Donut chart"],
    "ordinal":     ["100% Stacked bar", "Bar chart", "Horizontal bar", "Line chart"],
    "numeric":     ["Mean", "Histogram", "Box plot", "Violin plot"],
    "scale_7":     ["100% Stacked bar", "Box Stack", "Mean bar", "Bar chart", "Horizontal bar"],
    "scale_5":     ["100% Stacked bar", "Box Stack", "Mean bar", "Bar chart", "Horizontal bar"],
    "multi":       ["Horizontal bar", "Bar chart"],
    "grid":        ["Box Stack", "Mean bar", "Heatmap"],
    "open":        ["Word count bar"],
}

# Variable types where count/% toggle makes sense (not numeric, not open)
_PCT_TOGGLE_TYPES = {"categorical", "single", "ordinal", "scale_7", "scale_5", "multi", "grid"}

_SCALE_GROUPS = {
    "scale_7": [
        ("Bottom 2 (1–2)", [1, 2],    "#C0504D"),
        ("Middle (3–5)",   [3, 4, 5], "#FFD966"),
        ("Top 2 (6–7)",    [6, 7],    "#4472C4"),
    ],
    "scale_5": [
        ("Bottom 2 (1–2)", [1, 2], "#C0504D"),
        ("Middle (3)",     [3],    "#FFD966"),
        ("Top 2 (4–5)",   [4, 5], "#4472C4"),
    ],
}

# Qualitative palette for 100% stacked bar segments
_SEG_COLORS = px.colors.qualitative.Safe


_DEFAULT_COLOR = "#2E75B6"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_codebook(df):
    """Return codebook from store; auto-build from qnr_questions if absent."""
    cb = server_store.get_val("codebook")
    if cb:
        return cb
    questions = server_store.get_val("qnr_questions") or []
    if df is not None and questions:
        cb = build_codebook(df, questions)
        server_store.set_val("codebook", cb)
        return cb
    return None


def _invalidate_codebook():
    """Force codebook rebuild on next render (call after data/QNR changes)."""
    server_store.set_val("codebook", None)


def _get_df(state: dict):
    raw = server_store.get_df("raw_df")
    if raw is not None:
        return raw
    X = server_store.get_df("X_train")
    if X is not None:
        df = X.copy()
        y = server_store.get_df("y_train")
        target = (state or {}).get("target_col", "")
        if y is not None and target:
            df[target] = y.values
        return df
    return None


def _short(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _is_scale_grid(tile: dict) -> bool:
    return (tile.get("var_type") in ("scale_7", "scale_5")
            and len(tile.get("all_cols", [])) > 1)


# ── Label resolution ──────────────────────────────────────────────────────────

def _col_to_label(tile: dict, qnr_questions: list) -> dict:
    """Map dataset_col → display_label for multi/grid tiles."""
    all_cols = tile.get("all_cols", [])
    code = tile.get("code", "").upper()

    opts = []
    for q in (qnr_questions or []):
        if q.get("code", "").upper() == code:
            opts = q.get("options", [])
            break

    if not opts:
        raw_opts = tile.get("options", [])
        if raw_opts:
            if isinstance(raw_opts[0], dict):
                opts = [o.get("option_label", "") for o in raw_opts]
            elif isinstance(raw_opts[0], str):
                opts = raw_opts

    return {col: (opts[i] if i < len(opts) and opts[i] else col)
            for i, col in enumerate(all_cols)}


def _value_labels(tile: dict, qnr_questions: list) -> dict:
    """Map str(raw_value) → display_label for single-column questions."""
    labels = {str(k): str(v) for k, v in tile.get("value_labels", {}).items()}
    if labels:
        return labels

    raw_opts = tile.get("options", [])
    if raw_opts:
        if isinstance(raw_opts[0], dict):
            for i, opt in enumerate(raw_opts):
                vc = opt.get("value_coding", "").strip()
                ol = opt.get("option_label", "").strip()
                if "=" in vc:
                    for part in vc.split(","):
                        if "=" in part:
                            k, v = part.strip().split("=", 1)
                            labels.setdefault(k.strip(), v.strip())
                elif vc and ol:
                    labels.setdefault(vc, ol)
                elif ol:
                    labels.setdefault(str(i + 1), ol)
        elif isinstance(raw_opts[0], str):
            for i, opt in enumerate(raw_opts):
                if opt:
                    labels.setdefault(str(i + 1), opt)
        if labels:
            return labels

    code = tile.get("code", "").upper()
    for q in (qnr_questions or []):
        if q.get("code", "").upper() == code:
            for i, opt in enumerate(q.get("options", [])):
                if isinstance(opt, str) and opt:
                    labels.setdefault(str(i + 1), opt)
            break

    return labels


# ── Text helpers ──────────────────────────────────────────────────────────────

def _pct_text(vals):
    return [f"{v:.1f}%" for v in vals]


def _count_text(vals):
    return [str(int(v)) for v in vals]


# ── Renderers ─────────────────────────────────────────────────────────────────

def _render_numeric_mean(series, tile) -> go.Figure:
    """For continuous numeric variables: show mean prominently + summary stats."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numeric data", showarrow=False,
                           font=dict(size=13, color="gray"))
        return fig

    mean_v   = vals.mean()
    median_v = vals.median()
    std_v    = vals.std()
    n        = len(vals)

    fig = go.Figure(go.Indicator(
        mode="number",
        value=mean_v,
        number={"font": {"size": 52, "color": _DEFAULT_COLOR},
                "valueformat": ".2f"},
        title={"text": "Mean", "font": {"size": 13, "color": "#6b7280"}},
        domain={"x": [0.1, 0.9], "y": [0.35, 0.95]},
    ))
    stats = (f"Median: {median_v:.2f}  |  SD: {std_v:.2f}  |  N: {n}  |  "
             f"Min: {vals.min():.2f}  |  Max: {vals.max():.2f}")
    fig.add_annotation(text=stats, x=0.5, y=0.12, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=10, color="#6b7280"),
                       align="center")
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    return fig


def _render_100pct_stacked(series, tile, show_pct, qnr_questions) -> go.Figure:
    """
    Single horizontal 100% stacked bar — each response option is one colour segment.
    Used for single-select categorical/ordinal/scale questions (responses sum to 100%).
    """
    labels = _value_labels(tile, qnr_questions)
    mapped = series.dropna().map(lambda x: labels.get(str(x), str(x)))
    total  = len(mapped)
    if total == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False,
                           font=dict(size=13, color="gray"))
        return fig

    counts = mapped.value_counts().sort_index()

    fig = go.Figure()
    for i, (cat, cnt) in enumerate(counts.items()):
        pct  = cnt / total * 100
        text = f"{pct:.1f}%" if show_pct else str(int(cnt))
        fig.add_trace(go.Bar(
            name=_short(str(cat), 35),
            x=[pct],          # bar width always = pct so total = 100%
            y=[""],
            orientation="h",
            marker_color=_SEG_COLORS[i % len(_SEG_COLORS)],
            text=text,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10),
            hovertemplate=(
                f"<b>{cat}</b><br>Count: {cnt}<br>%: {pct:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False,
                   zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05,
                    xanchor="left", x=0, font=dict(size=9),
                    traceorder="normal"),
        margin=dict(t=5, b=5, l=5, r=5),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _render_scale_grid(df, tile, chart_type, qnr_questions) -> go.Figure:
    col_map     = _col_to_label(tile, qnr_questions)
    cols        = [c for c in tile.get("all_cols", []) if c in df.columns]
    full_labels = [col_map.get(c, c) for c in cols]
    bar_labels  = [_short(l) for l in full_labels]
    var_type    = tile.get("var_type", "scale_7")

    if chart_type == "Mean bar":
        means = [pd.to_numeric(df[c], errors="coerce").mean() for c in cols]
        fig = go.Figure(go.Bar(
            x=bar_labels, y=means, marker_color=_DEFAULT_COLOR,
            text=[f"{m:.2f}" for m in means], textposition="outside",
            customdata=full_labels,
            hovertemplate="<b>%{customdata}</b><br>Mean: %{y:.2f}<extra></extra>",
        ))
        sp = tile.get("scale_points", 7)
        fig.update_layout(yaxis_title="Mean rating",
                          yaxis=dict(range=[0, (sp or 7) + 0.5]))
        return fig

    if chart_type == "Heatmap":
        mat = df[cols].apply(pd.to_numeric, errors="coerce")
        mat.columns = bar_labels
        return px.imshow(mat.T, aspect="auto",
                         labels=dict(x="Respondent", y="Item", color="Score"),
                         color_continuous_scale="Blues")

    # Box Stack — default for scale grids
    groups = _SCALE_GROUPS.get(var_type, _SCALE_GROUPS["scale_7"])
    fig = go.Figure()
    for group_name, rating_vals, color in groups:
        pcts = []
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            total = len(s)
            pcts.append(s.isin(rating_vals).sum() / total * 100 if total else 0)
        fig.add_trace(go.Bar(
            name=group_name, x=bar_labels, y=pcts,
            marker_color=color,
            text=[f"{p:.0f}%" for p in pcts],
            textposition="inside",
            textfont=dict(color="white", size=9),
            customdata=full_labels,
            hovertemplate=(
                "<b>%{customdata}</b><br>" + group_name + ": %{y:.1f}%<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="% Respondents", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def _render_single(series, tile, chart_type, show_pct, qnr_questions,
                   breakout=None) -> go.Figure:
    labels = _value_labels(tile, qnr_questions)
    mapped = series.dropna().map(lambda x: labels.get(str(x), str(x)))
    total  = len(series.dropna())
    counts = mapped.value_counts().sort_index()
    x_lbl  = counts.index.astype(str).tolist()

    y_vals  = counts.values / total * 100 if show_pct else counts.values.astype(float)
    y_label = "% Respondents" if show_pct else "Count"
    texts   = _pct_text(y_vals) if show_pct else _count_text(y_vals)

    if chart_type == "Bar chart":
        if breakout is not None:
            tmp = pd.DataFrame({"val": mapped,
                                 "grp": breakout.reindex(mapped.index)}).dropna()
            if show_pct:
                grp = tmp.groupby(["grp", "val"]).size().reset_index(name="n")
                tots = tmp.groupby("grp").size().reset_index(name="tot")
                grp  = grp.merge(tots, on="grp")
                grp["pct"] = grp["n"] / grp["tot"] * 100
                fig = px.bar(grp, x="val", y="pct", color="grp", barmode="group",
                             labels={"val": "", "pct": "% Respondents", "grp": ""})
            else:
                grp = tmp.groupby(["grp", "val"]).size().reset_index(name="n")
                fig = px.bar(grp, x="val", y="n", color="grp", barmode="group",
                             labels={"val": "", "n": "Count", "grp": ""})
        else:
            fig = go.Figure(go.Bar(x=x_lbl, y=y_vals, marker_color=_DEFAULT_COLOR,
                                   text=texts, textposition="outside"))
            fig.update_layout(yaxis_title=y_label, showlegend=False)

    elif chart_type == "Horizontal bar":
        if breakout is not None:
            tmp = pd.DataFrame({"val": mapped,
                                 "grp": breakout.reindex(mapped.index)}).dropna()
            if show_pct:
                grp = tmp.groupby(["val", "grp"]).size().reset_index(name="n")
                tots = tmp.groupby("grp").size().reset_index(name="tot")
                grp  = grp.merge(tots, on="grp")
                grp["pct"] = grp["n"] / grp["tot"] * 100
                fig = px.bar(grp, y="val", x="pct", color="grp", barmode="group",
                             orientation="h",
                             labels={"val": "", "pct": "% Respondents", "grp": ""})
            else:
                grp = tmp.groupby(["val", "grp"]).size().reset_index(name="n")
                fig = px.bar(grp, y="val", x="n", color="grp", barmode="group",
                             orientation="h",
                             labels={"val": "", "n": "Count", "grp": ""})
        else:
            fig = go.Figure(go.Bar(y=x_lbl, x=y_vals, orientation="h",
                                   marker_color=_DEFAULT_COLOR,
                                   text=texts, textposition="outside"))
            fig.update_layout(xaxis_title=y_label,
                              yaxis=dict(autorange="reversed"), showlegend=False)

    elif chart_type == "Pie chart":
        fig = px.pie(values=y_vals, names=x_lbl)

    elif chart_type == "Donut chart":
        fig = px.pie(values=y_vals, names=x_lbl, hole=0.4)

    elif chart_type in ("Mean bar", "Mean line"):
        vals     = pd.to_numeric(series, errors="coerce").dropna()
        mean_val = vals.mean()
        fig = go.Figure(go.Bar(x=x_lbl, y=y_vals, marker_color=_DEFAULT_COLOR,
                               text=texts, textposition="outside"))
        fig.add_hline(
            y=mean_val / total * 100 if show_pct else mean_val,
            line_dash="dash", line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
        )
        scale_note = ""
        if tile.get("scale_low") and tile.get("scale_high"):
            sp = tile.get("scale_points", "")
            scale_note = f"1={tile['scale_low']} → {sp}={tile['scale_high']}"
        fig.update_layout(yaxis_title=y_label, xaxis_title=scale_note)

    elif chart_type == "Histogram":
        fig = px.histogram(pd.to_numeric(series, errors="coerce").dropna(),
                           histnorm="percent" if show_pct else "")

    elif chart_type == "Box plot":
        vals = pd.to_numeric(series, errors="coerce")
        if breakout is not None:
            tmp = pd.DataFrame({"val": vals, "grp": breakout}).dropna()
            fig = px.box(tmp, x="grp", y="val", labels={"grp": "", "val": ""})
        else:
            fig = px.box(y=vals.dropna())

    elif chart_type == "Violin plot":
        vals = pd.to_numeric(series, errors="coerce")
        if breakout is not None:
            tmp = pd.DataFrame({"val": vals, "grp": breakout}).dropna()
            fig = px.violin(tmp, x="grp", y="val", box=True,
                            labels={"grp": "", "val": ""})
        else:
            fig = px.violin(y=vals.dropna(), box=True)

    elif chart_type == "Line chart":
        fig = go.Figure(go.Scatter(x=x_lbl, y=y_vals, mode="lines+markers"))
        fig.update_layout(yaxis_title=y_label)

    elif chart_type == "Word count bar":
        words = series.dropna().astype(str).str.lower().str.split().explode()
        wc = words.value_counts().head(20)
        fig = px.bar(x=wc.values, y=wc.index, orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"))

    else:
        fig = go.Figure(go.Bar(x=x_lbl, y=y_vals))

    return fig


def _render_multi(df, tile, chart_type, show_pct, qnr_questions,
                  breakout=None) -> go.Figure:
    col_map     = _col_to_label(tile, qnr_questions)
    cols        = [c for c in tile.get("all_cols", []) if c in df.columns]
    full_labels = [col_map.get(c, c) for c in cols]
    bar_labels  = [_short(l) for l in full_labels]
    total       = len(df)

    if breakout is not None:
        grp_vals = sorted(breakout.dropna().unique())
        fig = go.Figure()
        for gv in grp_vals:
            mask = breakout == gv
            grp_total = mask.sum()
            raw = [pd.to_numeric(df.loc[mask, c], errors="coerce").sum() for c in cols]
            plot_vals = [v / grp_total * 100 for v in raw] if show_pct else raw
            texts = _pct_text(plot_vals) if show_pct else _count_text(plot_vals)
            if chart_type == "Bar chart":
                fig.add_trace(go.Bar(name=str(gv), x=bar_labels, y=plot_vals,
                                     text=texts, textposition="outside"))
            else:
                fig.add_trace(go.Bar(name=str(gv), y=bar_labels, x=plot_vals,
                                     orientation="h", text=texts,
                                     textposition="outside"))
        fig.update_layout(barmode="group")
    else:
        raw       = [pd.to_numeric(df[c], errors="coerce").sum() for c in cols]
        plot_vals = [v / total * 100 for v in raw] if show_pct else raw
        texts     = _pct_text(plot_vals) if show_pct else _count_text(plot_vals)
        axis_lbl  = "% Respondents" if show_pct else "Count"

        if chart_type == "Bar chart":
            fig = go.Figure(go.Bar(
                x=bar_labels, y=plot_vals, marker_color=_DEFAULT_COLOR,
                text=texts, textposition="outside",
                customdata=full_labels,
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f}<extra></extra>",
            ))
            fig.update_layout(yaxis_title=axis_lbl, showlegend=False)
        else:
            fig = go.Figure(go.Bar(
                y=bar_labels, x=plot_vals, orientation="h",
                marker_color=_DEFAULT_COLOR,
                text=texts, textposition="outside",
                customdata=full_labels,
                hovertemplate="<b>%{customdata}</b><br>%{x:.1f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title=axis_lbl,
                              yaxis=dict(autorange="reversed"), showlegend=False)

    fig.update_layout(height=max(280, len(cols) * 30 + 80))
    return fig


def _dispatch(df, tile, qnr_questions, show_pct=True,
              chart_type_override=None, breakout=None) -> go.Figure:
    var_type = tile.get("var_type", "categorical")
    g_type   = tile.get("group_type", "single")
    col      = tile.get("dataset_col", "")

    chart_opts = CHART_OPTIONS.get(var_type, ["Bar chart"])
    if chart_type_override and chart_type_override in chart_opts:
        chart_type = chart_type_override
    else:
        chart_type = tile.get("chart_type", chart_opts[0])
        if chart_type not in chart_opts:
            chart_type = chart_opts[0]

    is_scale     = var_type in ("scale_7", "scale_5")
    is_multi_col = len(tile.get("all_cols", [])) > 1

    # Multi-column scale / grid → Box Stack renderer
    if (is_scale and is_multi_col) or g_type == "grid":
        ct = chart_type if chart_type in ("Mean bar", "Heatmap") else "Box Stack"
        return _render_scale_grid(df, tile, ct, qnr_questions)

    # Multi-select → bar per option (not summing to 100%)
    # But first check: if values are actually scale ratings (1–7), treat as scale_7
    if g_type == "multi" and is_multi_col:
        try:
            cols = [c for c in tile.get("all_cols", []) if c in df.columns]
            sample = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if not sample.empty:
                max_val = sample.max().max()
                min_val = sample.min().min()
                if pd.notna(max_val) and min_val >= 1 and 2 < max_val <= 7:
                    effective_type = "scale_7" if max_val > 5 else "scale_5"
                    patched = dict(tile, var_type=effective_type)
                    ct = chart_type if chart_type in ("Mean bar", "Heatmap") else "Box Stack"
                    return _render_scale_grid(df, patched, ct, qnr_questions)
        except Exception:
            pass
        return _render_multi(df, tile, chart_type, show_pct, qnr_questions, breakout)

    if col and col in df.columns:
        # Continuous numeric → mean display
        if chart_type == "Mean":
            return _render_numeric_mean(df[col], tile)

        # Single-select categorical/ordinal/scale → 100% stacked bar
        if chart_type == "100% Stacked bar":
            return _render_100pct_stacked(df[col], tile, show_pct, qnr_questions)

        return _render_single(df[col], tile, chart_type, show_pct, qnr_questions, breakout)

    fig = go.Figure()
    fig.add_annotation(text="No data column mapped", showarrow=False,
                       font=dict(size=14, color="gray"))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout(state: dict) -> html.Div:
    df       = _get_df(state or {})
    codebook = _get_codebook(df)

    header = html.Div([
        html.H2("Stage 2.6 — Charts Portal"),
        html.P("Charts from raw data — axes use QNR option labels, not variable codes."),
    ], className="stage-header")

    if not codebook:
        return html.Div([header, dbc.Alert(
            "No variable mapping found. Complete the Variable Mapping stage first.",
            color="warning"
        )])

    if df is None:
        return html.Div([header, dbc.Alert(
            "No dataset loaded. Complete Stage 0 first.", color="warning"
        )])

    included  = [t for t in codebook
                 if t.get("include") and t.get("dataset_col")
                 and t["dataset_col"] in df.columns]
    all_types = sorted({t["var_type"] for t in included})
    breakout_opts = [{"label": "None", "value": "None"}] + [
        {"label": c, "value": c}
        for c in df.columns if 1 < df[c].nunique() <= 10
    ]

    return html.Div([
        header,
        dcc.Store(id="cp2c-overrides", data={}, storage_type="session"),
        dcc.Store(id="cp2c-raw-tile-code", storage_type="memory"),

        # Floating Apply button
        html.Div(
            dbc.Button(
                [html.I(className="bi bi-check-lg me-1"), "Apply"],
                id="cp2c-apply-btn",
                color="primary",
                n_clicks=0,
                title="Apply current filters and re-render charts",
            ),
            style={"position": "fixed", "top": "18px", "right": "36px",
                   "zIndex": 9999},
        ),

        # Raw-data modal (shared; populated on demand)
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="cp2c-raw-modal-title")),
            dbc.ModalBody(id="cp2c-raw-modal-body",
                          style={"maxHeight": "70vh", "overflowY": "auto"}),
        ], id="cp2c-raw-modal", size="xl", scrollable=True, is_open=False),

        # Controls
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Variable types", className="fw-semibold"),
                    dcc.Dropdown(
                        id="cp2c-types",
                        options=[{"label": t, "value": t} for t in all_types],
                        value=all_types, multi=True, clearable=False,
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Search", className="fw-semibold"),
                    dbc.Input(id="cp2c-search",
                              placeholder="code or keyword…", debounce=True),
                ], width=4),
                dbc.Col([
                    html.Label("Columns per row", className="fw-semibold"),
                    dcc.Slider(id="cp2c-cols", min=1, max=3, step=1, value=2,
                               marks={"1": "1", "2": "2", "3": "3"},
                               tooltip={"always_visible": False}),
                ], width=2),
                dbc.Col([
                    html.Label("Breakout by", className="fw-semibold"),
                    dcc.Dropdown(id="cp2c-breakout",
                                 options=breakout_opts, value="None",
                                 clearable=False),
                ], width=2),
            ], className="g-3"),
        ]), className="mb-3"),

        html.Div(id="cp2c-summary",
                 className="text-muted mb-2",
                 style={"fontSize": "0.85rem"}),
        html.Div(id="cp2c-charts-grid"),

        # Export
        dbc.Card(dbc.CardBody([
            html.H5("Export to PowerPoint"),
            html.P([
                html.I(className="bi bi-info-circle me-1"),
                "Set a Slide # on each chart card above. "
                "Charts sharing the same number appear on the same slide.",
            ], className="text-muted", style={"fontSize": "0.85rem"}),
            dbc.Button(
                [html.I(className="bi bi-file-earmark-ppt me-2"), "Generate PPT"],
                id="cp2c-export-btn", color="primary",
            ),
            html.Div(id="cp2c-export-status", className="mt-2"),
            dcc.Download(id="cp2c-download"),
        ]), className="mt-3"),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("cp2c-overrides", "data"),
    Input({"type": "cp2c-ct",       "index": ALL}, "value"),
    Input({"type": "cp2c-pct-tile", "index": ALL}, "value"),
    Input({"type": "cp2c-slide",    "index": ALL}, "value"),
    State({"type": "cp2c-ct",       "index": ALL}, "id"),
    State({"type": "cp2c-pct-tile", "index": ALL}, "id"),
    State({"type": "cp2c-slide",    "index": ALL}, "id"),
    State("cp2c-overrides", "data"),
    prevent_initial_call=True,
)
def update_overrides(ct_vals, pct_vals, slide_vals, ct_ids, pct_ids, slide_ids, current):
    """Persist per-tile chart type, count/%, and slide number selections."""
    overrides = dict(current or {})
    changed = False
    for ct_id, val in zip(ct_ids or [], ct_vals or []):
        key = ct_id["index"]
        if val and overrides.get(key) != val:
            overrides[key] = val
            changed = True
    for pct_id, val in zip(pct_ids or [], pct_vals or []):
        key = pct_id["index"] + "__pct"
        if val and overrides.get(key) != val:
            overrides[key] = val
            changed = True
    for slide_id, val in zip(slide_ids or [], slide_vals or []):
        key = slide_id["index"] + "__slide"
        cur = overrides.get(key)
        # Store the slide number (or clear it if blank)
        new_val = int(val) if val not in (None, "") else None
        if new_val != cur:
            if new_val is None:
                overrides.pop(key, None)
            else:
                overrides[key] = new_val
            changed = True
    return overrides if changed else no_update


@callback(
    Output("cp2c-charts-grid", "children"),
    Output("cp2c-summary",     "children"),
    Input("cp2c-apply-btn",  "n_clicks"),
    State("cp2c-types",     "value"),
    State("cp2c-search",    "value"),
    State("cp2c-cols",      "value"),
    State("cp2c-breakout",  "value"),
    State("cp2c-overrides", "data"),
    State("app-state",      "data"),
)
def render_charts(n_clicks, type_filter, search, cols_per_row, breakout_col, overrides, state):
    df            = _get_df(state or {})
    codebook      = _get_codebook(df)
    qnr_questions = server_store.get_val("qnr_questions") or []

    if not codebook or df is None:
        return dbc.Alert("Complete Variable Mapping first.", color="info"), ""

    cols_per_row = int(cols_per_row or 2)
    type_filter  = type_filter or []
    overrides    = overrides or {}

    included = [t for t in codebook
                if t.get("include") and t.get("dataset_col")
                and t["dataset_col"] in df.columns]

    visible = [
        t for t in included
        if t["var_type"] in type_filter
        and (not search
             or search.lower() in t["code"].lower()
             or search.lower() in t.get("question", "").lower())
    ]

    summary = f"Showing {len(visible)} of {len(included)} variables"
    if breakout_col and breakout_col != "None":
        summary += f"  |  Breakout: {breakout_col}"

    if not visible:
        return dbc.Alert("No variables match the current filters.", color="info"), summary

    breakout_series = (df[breakout_col]
                       if breakout_col and breakout_col != "None"
                       and breakout_col in df.columns
                       else None)

    chart_rows = []
    row_cols   = []

    for tile in visible:
        code       = tile["code"]
        q_text     = tile.get("question", "").strip()
        var_type   = tile.get("var_type", "categorical")
        chart_opts = CHART_OPTIONS.get(var_type, ["Bar chart"])

        # Default chart type
        if _is_scale_grid(tile) or tile.get("group_type") == "grid":
            default_ct = "Box Stack"
        else:
            default_ct = chart_opts[0]   # first option is the recommended default

        chosen_ct = overrides.get(code, default_ct)
        if chosen_ct not in chart_opts:
            chosen_ct = default_ct

        # Per-tile show_pct (default: % for non-numeric, n/a for numeric)
        show_pct_default = "pct" if var_type != "numeric" else "count"
        show_pct = (overrides.get(f"{code}__pct", show_pct_default) == "pct")

        scale_note = ""
        if tile.get("scale_low") and tile.get("scale_high"):
            sp = tile.get("scale_points", "")
            scale_note = f"1 = {tile['scale_low']}  →  {sp} = {tile['scale_high']}"

        try:
            fig = _dispatch(df, tile, qnr_questions, show_pct,
                            chart_type_override=chosen_ct,
                            breakout=breakout_series)

            # Numeric mean and 100% stacked bar are shorter
            if var_type == "numeric" and chosen_ct == "Mean":
                h = 220
            elif chosen_ct == "100% Stacked bar":
                h = 190
            elif _is_scale_grid(tile) or tile.get("group_type") == "grid":
                h = 360
            else:
                h = 320

            fig.update_layout(
                height=h,
                margin=dict(t=30, b=10, l=30, r=20),
                font=dict(size=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            graph = dcc.Graph(figure=fig,
                              config={"displayModeBar": False},
                              style={"height": f"{h + 10}px"})
        except Exception as exc:
            graph = dbc.Alert(f"Chart error: {exc}", color="danger")

        # Per-tile controls row
        controls = [
            dbc.RadioItems(
                id={"type": "cp2c-ct", "index": code},
                options=[{"label": o, "value": o} for o in chart_opts],
                value=chosen_ct,
                inline=True,
                style={"fontSize": "0.72rem"},
            ),
        ]
        # Count/% toggle — hide for numeric (mean display)
        if var_type in _PCT_TOGGLE_TYPES:
            controls.append(
                dbc.RadioItems(
                    id={"type": "cp2c-pct-tile", "index": code},
                    options=[{"label": "Count", "value": "count"},
                             {"label": "%",     "value": "pct"}],
                    value="pct" if show_pct else "count",
                    inline=True,
                    style={"fontSize": "0.72rem", "marginLeft": "12px"},
                )
            )
        # Slide number input
        controls.append(
            html.Div([
                html.Small("Slide #", style={"color": "#6b7280", "marginRight": "4px"}),
                dbc.Input(
                    id={"type": "cp2c-slide", "index": code},
                    type="number", min=1, step=1,
                    placeholder="—",
                    value=overrides.get(code + "__slide"),
                    size="sm",
                    style={"width": "64px", "display": "inline-block",
                           "padding": "1px 4px", "fontSize": "0.72rem"},
                ),
            ], style={"marginLeft": "auto", "display": "flex",
                      "alignItems": "center", "whiteSpace": "nowrap"})
        )

        card = dbc.Card(dbc.CardBody([
            html.Div([
                html.Div([
                    html.Strong(code),
                    html.Span(f" — {q_text}",
                              style={"fontSize": "0.82rem", "color": "#374151"}),
                ]),
                html.Button(
                    [html.I(className="bi bi-table me-1"), "Data"],
                    id={"type": "cp2c-raw-btn", "index": code},
                    title="View raw data used for this chart",
                    n_clicks=0,
                    style={"fontSize": "0.72rem", "padding": "2px 8px",
                           "borderRadius": "4px", "border": "1px solid #d1d5db",
                           "background": "#f9fafb", "color": "#374151",
                           "cursor": "pointer", "whiteSpace": "nowrap",
                           "flexShrink": "0"},
                ),
            ], className="d-flex justify-content-between align-items-start mb-1"),
            html.Small(scale_note, className="text-muted d-block mb-1")
            if scale_note else None,
            html.Div(controls, className="d-flex align-items-center flex-wrap mb-1"),
            graph,
        ]), className="mb-3 h-100")

        row_cols.append(dbc.Col(card, width=12 // cols_per_row))
        if len(row_cols) >= cols_per_row:
            chart_rows.append(dbc.Row(row_cols, className="mb-1"))
            row_cols = []

    if row_cols:
        chart_rows.append(dbc.Row(row_cols))

    return html.Div(chart_rows), summary


@callback(
    Output("cp2c-download",      "data"),
    Output("cp2c-export-status", "children"),
    Input("cp2c-export-btn",     "n_clicks"),
    State("cp2c-types",    "value"),
    State("cp2c-search",   "value"),
    State("cp2c-overrides","data"),
    State("app-state",     "data"),
    prevent_initial_call=True,
)
def export_ppt(n_clicks, type_filter, search, overrides, state):
    if not n_clicks:
        return no_update, no_update

    df            = _get_df(state or {})
    codebook      = _get_codebook(df)
    qnr_questions = server_store.get_val("qnr_questions") or []

    if not codebook or df is None:
        return no_update, dbc.Alert("No codebook/data available.", color="warning")

    type_filter = type_filter or []
    overrides   = overrides or {}
    included    = [t for t in codebook
                   if t.get("include") and t.get("dataset_col")
                   and t["dataset_col"] in df.columns]
    visible     = [
        t for t in included
        if t["var_type"] in type_filter
        and (not search
             or search.lower() in t["code"].lower()
             or search.lower() in t.get("question", "").lower())
    ]

    if not visible:
        return no_update, dbc.Alert("No charts visible to export.", color="warning")

    try:
        buf = build_pptx(visible, df, overrides, qnr_questions)
        return (
            dcc.send_bytes(buf.read(), "charts_portal.pptx"),
            dbc.Alert("PPT ready — downloading…", color="success", duration=3000),
        )
    except Exception as exc:
        return no_update, dbc.Alert(
            f"PPT export error: {exc}. Install: pip install python-pptx",
            color="danger",
        )


@callback(
    Output("cp2c-raw-tile-code", "data"),
    Output("cp2c-raw-modal",     "is_open"),
    Input({"type": "cp2c-raw-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_raw_modal(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in (n_clicks_list or [])):
        return no_update, no_update
    return triggered["index"], True


@callback(
    Output("cp2c-raw-modal-title", "children"),
    Output("cp2c-raw-modal-body",  "children"),
    Input("cp2c-raw-tile-code", "data"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def populate_raw_modal(code, state):
    if not code:
        return "Raw Data", ""

    df            = _get_df(state or {})
    codebook      = _get_codebook(df)
    qnr_questions = server_store.get_val("qnr_questions") or []

    if df is None or not codebook:
        return code, dbc.Alert("No data available.", color="warning")

    tile = next((t for t in codebook if t["code"] == code), None)
    if not tile:
        return code, dbc.Alert("Question not found in codebook.", color="warning")

    all_cols = [c for c in tile.get("all_cols", []) if c in df.columns]
    if not all_cols:
        return code, dbc.Alert("No data columns mapped for this question.", color="warning")

    raw = df[all_cols].copy()

    # Apply value labels for single-column questions
    if len(all_cols) == 1:
        labels = _value_labels(tile, qnr_questions)
        if labels:
            col = all_cols[0]
            raw[col] = raw[col].map(lambda x: labels.get(str(x), x))

    # Rename columns using option labels for multi-column questions
    col_map = _col_to_label(tile, qnr_questions)
    raw = raw.rename(columns={c: col_map.get(c, c) for c in all_cols})

    n_rows   = len(raw)
    preview  = raw.head(500)
    title    = f"{code} — {tile.get('question', '')[:80]}"
    summary  = html.P(
        f"Showing up to 500 of {n_rows} rows · {len(all_cols)} column(s)",
        className="text-muted mb-2",
        style={"fontSize": "0.8rem"},
    )

    table = dash_table.DataTable(
        data=preview.to_dict("records"),
        columns=[{"name": c, "id": c} for c in preview.columns],
        page_size=25,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "0.8rem", "padding": "4px 8px",
                    "textAlign": "left", "maxWidth": "220px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"fontWeight": "600", "background": "#f3f4f6",
                      "fontSize": "0.78rem", "borderBottom": "2px solid #d1d5db"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": "#f9fafb"},
        ],
    )

    return title, [summary, table]
