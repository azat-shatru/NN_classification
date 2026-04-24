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
    "scale_7":     ["100% Stacked bar", "Box Stack", "Box Stack H", "Mean bar", "Bar chart", "Horizontal bar"],
    "scale_5":     ["100% Stacked bar", "Box Stack", "Box Stack H", "Mean bar", "Bar chart", "Horizontal bar"],
    "multi":       ["Horizontal bar", "Bar chart"],
    "grid":        ["Box Stack", "Box Stack H", "Mean bar", "Heatmap"],
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
    bar_labels  = [_short(l) for l in full_labels]   # short for vertical
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

    # ── Box Stack (vertical) or Box Stack H (horizontal) ─────────────────
    horizontal = (chart_type == "Box Stack H")
    # Use full labels (no truncation) when horizontal — there's enough room
    labels_for_axis = full_labels if horizontal else bar_labels
    groups = _SCALE_GROUPS.get(var_type, _SCALE_GROUPS["scale_7"])

    fig = go.Figure()
    for group_name, rating_vals, color in groups:
        pcts = []
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            total = len(s)
            pcts.append(s.isin(rating_vals).sum() / total * 100 if total else 0)

        if horizontal:
            fig.add_trace(go.Bar(
                name=group_name,
                y=labels_for_axis, x=pcts,
                orientation="h",
                marker_color=color,
                text=[f"{p:.0f}%" for p in pcts],
                textposition="inside",
                textfont=dict(color="white", size=9),
                customdata=labels_for_axis,
                hovertemplate=(
                    "<b>%{customdata}</b><br>" + group_name + ": %{x:.1f}%<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Bar(
                name=group_name,
                x=labels_for_axis, y=pcts,
                marker_color=color,
                text=[f"{p:.0f}%" for p in pcts],
                textposition="inside",
                textfont=dict(color="white", size=9),
                customdata=full_labels,
                hovertemplate=(
                    "<b>%{customdata}</b><br>" + group_name + ": %{y:.1f}%<extra></extra>"
                ),
            ))

    if horizontal:
        fig.update_layout(
            barmode="stack",
            xaxis=dict(title="% Respondents", range=[0, 105]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1),
        )
    else:
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
        ct = chart_type if chart_type in ("Mean bar", "Heatmap", "Box Stack H") else "Box Stack"
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
        html.P("Select a question in the left panel. Ribbon settings apply only to that question."),
    ], className="stage-header")

    if not codebook:
        return html.Div([header, dbc.Alert(
            "No variable mapping found. Complete the Variable Mapping stage first.",
            color="warning",
        )])
    if df is None:
        return html.Div([header, dbc.Alert(
            "No dataset loaded. Complete Stage 0 first.", color="warning",
        )])

    included = [t for t in codebook
                if t.get("include") and t.get("dataset_col")
                and t["dataset_col"] in df.columns]
    if not included:
        return html.Div([header, dbc.Alert("No includeable variables found.", color="warning")])

    all_types     = sorted({t["var_type"] for t in included})
    init_code     = included[0]["code"]
    breakout_opts = [{"label": "None", "value": "None"}] + [
        {"label": c, "value": c}
        for c in df.columns if 1 < df[c].nunique() <= 10
    ]

    _sep = html.Div(style={
        "width": "1px", "height": "22px",
        "background": "#d1d5db", "margin": "0 10px", "flexShrink": "0",
    })

    return html.Div([
        header,
        dcc.Store(id="cp2c-overrides",      data={},      storage_type="session"),
        dcc.Store(id="cp2c-selected-code",  data=init_code, storage_type="memory"),
        dcc.Download(id="cp2c-download"),

        # Raw-data modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="cp2c-raw-modal-title")),
            dbc.ModalBody(id="cp2c-raw-modal-body",
                          style={"maxHeight": "70vh", "overflowY": "auto"}),
        ], id="cp2c-raw-modal", size="xl", scrollable=True, is_open=False),

        # ── Main content: left nav + right panel ──────────────────────────
        html.Div([

            # ── LEFT NAV PANEL ────────────────────────────────────────────
            html.Div([
                # Filter controls
                html.Div([
                    dcc.Dropdown(
                        id="cp2c-types",
                        options=[{"label": t, "value": t} for t in all_types],
                        value=all_types, multi=True, clearable=False,
                        placeholder="Variable types…",
                        style={"fontSize": "0.76rem"},
                    ),
                    dbc.Input(
                        id="cp2c-search",
                        placeholder="Search questions…",
                        debounce=True, size="sm",
                        style={"fontSize": "0.76rem", "marginTop": "6px"},
                    ),
                ], style={"padding": "10px 10px 6px"}),
                html.Hr(style={"margin": "0", "borderColor": "#e5e7eb"}),
                # Question list — populated by callback
                html.Div(id="cp2c-nav-list"),
            ], style={
                "width": "230px", "minWidth": "230px", "flexShrink": "0",
                "borderRight": "1px solid #e5e7eb",
                "height": "calc(100vh - 130px)",
                "overflowY": "auto",
                "backgroundColor": "#f8f9fa",
                "position": "sticky", "top": "0", "alignSelf": "flex-start",
            }),

            # ── RIGHT: ribbon + chart ─────────────────────────────────────
            html.Div([

                # Ribbon
                html.Div([
                    # Chart type
                    html.Div([
                        html.Small("Chart:", style={
                            "color": "#6b7280", "fontWeight": "600",
                            "marginRight": "6px", "whiteSpace": "nowrap",
                        }),
                        dbc.RadioItems(
                            id="cp2c-ribbon-ct",
                            options=[], value=None, inline=True,
                            style={"fontSize": "0.76rem"},
                        ),
                    ], className="d-flex align-items-center"),
                    _sep,
                    # Count / %
                    html.Div([
                        html.Small("Show:", style={
                            "color": "#6b7280", "fontWeight": "600",
                            "marginRight": "6px", "whiteSpace": "nowrap",
                        }),
                        dbc.RadioItems(
                            id="cp2c-ribbon-pct",
                            options=[{"label": "Count", "value": "count"},
                                     {"label": "%",     "value": "pct"}],
                            value="pct", inline=True,
                            style={"fontSize": "0.76rem"},
                        ),
                    ], id="cp2c-ribbon-pct-wrap", className="d-flex align-items-center"),
                    _sep,
                    # Slide #
                    html.Div([
                        html.Small("Slide #:", style={
                            "color": "#6b7280", "fontWeight": "600",
                            "marginRight": "4px", "whiteSpace": "nowrap",
                        }),
                        dbc.Input(
                            id="cp2c-ribbon-slide",
                            type="number", min=1, step=1,
                            placeholder="—", size="sm", debounce=True,
                            style={"width": "62px", "fontSize": "0.76rem"},
                        ),
                    ], className="d-flex align-items-center"),
                    _sep,
                    # Breakout
                    html.Div([
                        html.Small("Breakout:", style={
                            "color": "#6b7280", "fontWeight": "600",
                            "marginRight": "4px", "whiteSpace": "nowrap",
                        }),
                        dcc.Dropdown(
                            id="cp2c-breakout",
                            options=breakout_opts, value="None",
                            clearable=False,
                            style={"width": "130px", "fontSize": "0.76rem"},
                        ),
                    ], className="d-flex align-items-center"),
                    # Push remaining items right
                    html.Div(style={"flex": "1"}),
                    dbc.Button(
                        [html.I(className="bi bi-table me-1"), "Data"],
                        id="cp2c-raw-btn", color="light", size="sm",
                        className="me-2",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-file-earmark-ppt me-1"), "Export PPT"],
                        id="cp2c-export-btn", color="primary", size="sm",
                    ),
                    html.Div(id="cp2c-export-status", style={"marginLeft": "8px"}),
                ], style={
                    "display": "flex", "alignItems": "center", "flexWrap": "wrap",
                    "padding": "8px 16px", "gap": "4px",
                    "borderBottom": "1px solid #e5e7eb",
                    "backgroundColor": "white",
                    "position": "sticky", "top": "0", "zIndex": 100,
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                }),

                # Chart display area
                html.Div(id="cp2c-chart-area",
                         style={"padding": "20px 28px", "minHeight": "400px"}),

            ], style={
                "flex": "1", "minWidth": "0",
                "display": "flex", "flexDirection": "column",
                "backgroundColor": "white",
            }),
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "marginTop": "8px",
        }),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── 1. Nav click → update selected code ──────────────────────────────────────
@callback(
    Output("cp2c-selected-code", "data"),
    Input({"type": "cp2c-nav-item", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_question(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in (n_clicks_list or []) if n):
        return no_update
    return triggered["index"]


# ── 2. Main render: nav + ribbon + chart ──────────────────────────────────────
@callback(
    Output("cp2c-chart-area",        "children"),
    Output("cp2c-nav-list",          "children"),
    Output("cp2c-ribbon-ct",         "options"),
    Output("cp2c-ribbon-ct",         "value"),
    Output("cp2c-ribbon-pct",        "value"),
    Output("cp2c-ribbon-pct-wrap",   "style"),
    Output("cp2c-ribbon-slide",      "value"),

    Input("cp2c-selected-code",  "data"),
    Input("cp2c-overrides",      "data"),
    Input("cp2c-types",          "value"),
    Input("cp2c-search",         "value"),
    Input("cp2c-breakout",       "value"),

    State("app-state", "data"),
)
def render_view(selected_code, overrides, type_filter, search, breakout_col, state):
    df            = _get_df(state or {})
    codebook      = _get_codebook(df)
    qnr_questions = server_store.get_val("qnr_questions") or []
    overrides     = overrides or {}

    _empty_ribbon = ([], None, "pct", {}, None)

    if not codebook or df is None:
        return (dbc.Alert("Complete Variable Mapping and load data first.", color="info"),
                [], *_empty_ribbon)

    type_filter = type_filter or []
    included = [t for t in codebook
                if t.get("include") and t.get("dataset_col")
                and t["dataset_col"] in df.columns]
    visible  = [
        t for t in included
        if t["var_type"] in type_filter
        and (not search
             or search.lower() in t["code"].lower()
             or search.lower() in t.get("question", "").lower())
    ]

    # ── Build nav list ────────────────────────────────────────────────────
    # If selected code is no longer visible, pick the first visible
    visible_codes = {t["code"] for t in visible}
    if not selected_code or selected_code not in visible_codes:
        selected_code = visible[0]["code"] if visible else None

    nav_items = []
    for tile in visible:
        code      = tile["code"]
        q_short   = tile.get("question", "")[:50]
        is_active = (code == selected_code)
        nav_items.append(
            html.Div([
                html.Span(code, style={
                    "fontWeight": "700" if is_active else "600",
                    "fontSize": "0.78rem",
                    "color": "#1e3a8a" if is_active else "#374151",
                }),
                html.Div(q_short, style={
                    "fontSize": "0.69rem", "color": "#6b7280",
                    "marginTop": "1px", "lineHeight": "1.25",
                }),
            ],
            id={"type": "cp2c-nav-item", "index": code},
            n_clicks=0,
            style={
                "padding": "7px 10px 7px 12px",
                "cursor": "pointer",
                "borderLeft": f"3px solid {'#3a6df0' if is_active else 'transparent'}",
                "backgroundColor": "#e8edfb" if is_active else "transparent",
                "userSelect": "none",
            })
        )
    nav_div = html.Div(nav_items) if nav_items else html.Small(
        "No questions match filters.", className="text-muted d-block p-2")

    if not visible or not selected_code:
        return (dbc.Alert("No questions match the current filters.", color="info"),
                nav_div, *_empty_ribbon)

    # ── Resolve tile ──────────────────────────────────────────────────────
    tile = next((t for t in visible if t["code"] == selected_code), None)
    if not tile:
        return (dbc.Alert("Select a question.", color="info"), nav_div, *_empty_ribbon)

    var_type   = tile.get("var_type", "categorical")
    chart_opts = CHART_OPTIONS.get(var_type, ["Bar chart"])

    if _is_scale_grid(tile) or tile.get("group_type") == "grid":
        default_ct = "Box Stack"
    else:
        default_ct = chart_opts[0]

    chosen_ct = overrides.get(selected_code, default_ct)
    if chosen_ct not in chart_opts:
        chosen_ct = default_ct

    default_pct = "pct" if var_type != "numeric" else "count"
    pct_val     = overrides.get(f"{selected_code}__pct", default_pct)
    slide_val   = overrides.get(f"{selected_code}__slide")
    show_pct    = (pct_val == "pct")

    pct_style   = {} if var_type in _PCT_TOGGLE_TYPES else {"display": "none"}

    # ── Render chart ──────────────────────────────────────────────────────
    breakout_series = (
        df[breakout_col]
        if breakout_col and breakout_col != "None" and breakout_col in df.columns
        else None
    )
    scale_note = ""
    if tile.get("scale_low") and tile.get("scale_high"):
        sp = tile.get("scale_points", "")
        scale_note = f"1 = {tile['scale_low']}  →  {sp} = {tile['scale_high']}"

    try:
        fig = _dispatch(df, tile, qnr_questions, show_pct,
                        chart_type_override=chosen_ct,
                        breakout=breakout_series)
        n_items = len(tile.get("all_cols", []))
        if var_type == "numeric" and chosen_ct == "Mean":
            h = 300
        elif chosen_ct == "100% Stacked bar":
            h = 260
        elif chosen_ct == "Box Stack H":
            # Horizontal: auto-height per item so labels always fit
            h = max(380, n_items * 38 + 120)
        elif _is_scale_grid(tile) or tile.get("group_type") == "grid":
            h = max(420, n_items * 40 + 100)
        elif tile.get("group_type") == "multi":
            h = max(380, n_items * 34 + 80)
        else:
            h = 500

        fig.update_layout(
            height=h,
            margin=dict(t=30, b=40, l=50, r=30),
            font=dict(size=11),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        graph = dcc.Graph(
            figure=fig,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"height": f"{h + 10}px"},
        )
    except Exception as exc:
        graph = dbc.Alert(f"Chart error: {exc}", color="danger")

    q_text_full = tile.get("question", "")
    chart_area  = html.Div([
        # Question title
        html.Div([
            html.Span(selected_code, style={
                "fontWeight": "700", "color": "#1e3a8a",
                "fontSize": "1rem", "marginRight": "8px",
            }),
            html.Span(q_text_full, style={"fontSize": "0.9rem", "color": "#374151"}),
            html.Small(f" [{var_type}]", style={"color": "#9ca3af", "marginLeft": "6px"}),
            html.Div(scale_note, style={
                "fontSize": "0.76rem", "color": "#6b7280", "marginTop": "3px",
            }) if scale_note else None,
        ], style={
            "marginBottom": "12px",
            "borderBottom": "1px solid #e5e7eb",
            "paddingBottom": "8px",
        }),
        graph,
    ])

    ribbon_opts = [{"label": o, "value": o} for o in chart_opts]
    return chart_area, nav_div, ribbon_opts, chosen_ct, pct_val, pct_style, slide_val


# ── 3. Save ribbon settings to overrides ─────────────────────────────────────
@callback(
    Output("cp2c-overrides", "data"),
    Input("cp2c-ribbon-ct",    "value"),
    Input("cp2c-ribbon-pct",   "value"),
    Input("cp2c-ribbon-slide", "value"),
    State("cp2c-selected-code", "data"),
    State("cp2c-overrides",    "data"),
    prevent_initial_call=True,
)
def update_overrides_from_ribbon(ct_val, pct_val, slide_val, selected_code, current):
    if not selected_code:
        return no_update
    overrides = dict(current or {})
    changed   = False

    if ct_val and overrides.get(selected_code) != ct_val:
        overrides[selected_code] = ct_val
        changed = True

    if pct_val and overrides.get(f"{selected_code}__pct") != pct_val:
        overrides[f"{selected_code}__pct"] = pct_val
        changed = True

    new_slide = int(slide_val) if slide_val not in (None, "") else None
    cur_slide = overrides.get(f"{selected_code}__slide")
    if new_slide != cur_slide:
        if new_slide is None:
            overrides.pop(f"{selected_code}__slide", None)
        else:
            overrides[f"{selected_code}__slide"] = new_slide
        changed = True

    return overrides if changed else no_update


# ── 4. Export PPT ─────────────────────────────────────────────────────────────
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
        return no_update, dbc.Alert(f"PPT export error: {exc}", color="danger")


# ── 5. Raw data modal ─────────────────────────────────────────────────────────
@callback(
    Output("cp2c-raw-modal", "is_open"),
    Input("cp2c-raw-btn",    "n_clicks"),
    State("cp2c-raw-modal",  "is_open"),
    prevent_initial_call=True,
)
def toggle_raw_modal(n_clicks, is_open):
    return not is_open if n_clicks else is_open


@callback(
    Output("cp2c-raw-modal-title", "children"),
    Output("cp2c-raw-modal-body",  "children"),
    Input("cp2c-raw-modal",       "is_open"),
    State("cp2c-selected-code",   "data"),
    State("app-state",            "data"),
    prevent_initial_call=True,
)
def populate_raw_modal(is_open, code, state):
    if not is_open or not code:
        return no_update, no_update

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
        return code, dbc.Alert("No data columns mapped.", color="warning")

    raw = df[all_cols].copy()
    if len(all_cols) == 1:
        labels = _value_labels(tile, qnr_questions)
        if labels:
            raw[all_cols[0]] = raw[all_cols[0]].map(lambda x: labels.get(str(x), x))

    col_map = _col_to_label(tile, qnr_questions)
    raw     = raw.rename(columns={c: col_map.get(c, c) for c in all_cols})
    n_rows  = len(raw)
    preview = raw.head(500)
    title   = f"{code} — {tile.get('question', '')[:80]}"

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
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": "#f9fafb"}],
    )

    return title, [
        html.P(f"Showing up to 500 of {n_rows} rows · {len(all_cols)} column(s)",
               className="text-muted mb-2", style={"fontSize": "0.8rem"}),
        table,
    ]
