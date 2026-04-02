"""
Stage 2.5 — Variable Mapping & Visualisation Dashboard

Flow:
  1. Upload questionnaire doc → auto-parse → codebook draft
  2. Upload dataset (or reuse from Stage 0)
  3. Map dataset columns → question codes
  4. Edit any tile (question text, type, labels)
  5. Select chart type per tile
  6. View dashboard in Grid or Slide mode
  7. Export all charts as PDF
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
import tempfile, os
from pathlib import Path

from utils.state import init
from utils.ui import stage_header
from utils.qnr_parser import parse_questionnaire
from utils.col_mapper import build_codebook, group_columns


# ── chart types available per variable type ───────────────────────────────────
CHART_OPTIONS = {
    "categorical": ["Bar chart", "Pie chart", "Donut chart"],
    "ordinal":     ["Bar chart", "Horizontal bar", "Stacked bar", "Line chart"],
    "numeric":     ["Histogram", "Box plot", "Violin plot", "Line chart"],
    "scale_7":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "scale_5":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "multi":       ["Bar chart", "Horizontal bar"],
    "grid":        ["Heatmap", "Grouped bar"],
    "open":        ["Word count bar"],
    "single":      ["Bar chart", "Pie chart"],
}

VAR_TYPES = ["categorical", "ordinal", "numeric", "scale_7", "scale_5",
             "multi", "grid", "open", "single"]

Q_TYPE_TO_VAR = {
    "scale_7": "scale_7", "scale_5": "scale_5",
    "numeric": "numeric", "open": "open",
    "multi":   "multi",   "grid": "grid",
    "single":  "categorical",
}


# ── chart rendering ───────────────────────────────────────────────────────────

def _render_grid_chart(df: pd.DataFrame, tile: dict) -> go.Figure:
    """Render heatmap or grouped bar for grid questions (multiple row columns)."""
    cols = [c for c in tile.get("all_cols", [tile["dataset_col"]]) if c in df.columns]
    # Build mean per column
    means = {c: pd.to_numeric(df[c], errors="coerce").mean() for c in cols}
    labels = tile.get("row_labels", {})
    x_labels = [labels.get(c, c.split("_r")[-1] if "_r" in c else c) for c in cols]

    if tile.get("chart_type") == "Heatmap":
        mat = df[cols].apply(pd.to_numeric, errors="coerce")
        fig = px.imshow(mat.T, aspect="auto",
                        labels=dict(x="Respondent", y="Item", color="Score"),
                        title=tile["code"],
                        color_continuous_scale="Blues")
    else:
        fig = go.Figure(go.Bar(
            x=x_labels,
            y=[means[c] for c in cols],
            marker_color="#2E75B6",
            text=[f"{means[c]:.1f}" for c in cols],
            textposition="outside",
        ))
        fig.update_layout(title=tile["code"],
                          yaxis_title="Mean score",
                          xaxis_title="")

    fig.update_layout(height=300, margin=dict(t=40, b=40, l=20, r=20),
                      font=dict(size=10), title_font_size=13)
    return fig


def _render_multi_chart(df: pd.DataFrame, tile: dict) -> go.Figure:
    """Render bar chart for multi-select questions (binary 0/1 per option)."""
    cols = [c for c in tile.get("all_cols", [tile["dataset_col"]]) if c in df.columns]
    labels = tile.get("row_labels", {})
    x_labels = [labels.get(c, c) for c in cols]
    counts = [pd.to_numeric(df[c], errors="coerce").sum() for c in cols]
    pcts   = [v / len(df) * 100 for v in counts]

    fig = go.Figure(go.Bar(
        x=pcts, y=x_labels, orientation="h",
        marker_color="#2E75B6",
        text=[f"{p:.0f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=tile["code"], xaxis_title="% selected",
        yaxis=dict(autorange="reversed"),
        height=max(250, len(cols) * 30 + 60),
        margin=dict(t=40, b=20, l=20, r=20),
        font=dict(size=10), title_font_size=13,
    )
    return fig


def _render_chart(series: pd.Series, tile: dict, df: pd.DataFrame) -> go.Figure:
    chart   = tile.get("chart_type", "Bar chart")
    q_type  = tile.get("var_type", "categorical")
    title   = tile.get("code", "")
    labels  = tile.get("value_labels", {})

    # Apply value labels
    if labels:
        series = series.map(lambda x: labels.get(str(x), x))

    counts = series.value_counts().sort_index()

    if chart in ("Bar chart", "Single"):
        fig = px.bar(x=counts.index.astype(str), y=counts.values,
                     labels={"x": "", "y": "Count"},
                     title=title, color=counts.index.astype(str))
        fig.update_layout(showlegend=False)

    elif chart == "Horizontal bar":
        fig = px.bar(x=counts.values, y=counts.index.astype(str),
                     orientation="h",
                     labels={"x": "Count", "y": ""},
                     title=title)
        fig.update_layout(yaxis=dict(autorange="reversed"))

    elif chart == "Pie chart":
        fig = px.pie(values=counts.values, names=counts.index.astype(str),
                     title=title)

    elif chart == "Donut chart":
        fig = px.pie(values=counts.values, names=counts.index.astype(str),
                     hole=0.4, title=title)

    elif chart == "Histogram":
        fig = px.histogram(series.dropna(), x=series.name or "value",
                           title=title, nbins=20)

    elif chart == "Box plot":
        fig = px.box(y=series.dropna(), title=title)

    elif chart == "Violin plot":
        fig = px.violin(y=series.dropna(), title=title, box=True)

    elif chart == "Stacked bar":
        if tile.get("group_col") and tile["group_col"] in df.columns:
            grp = df.groupby([series.name, tile["group_col"]]).size().reset_index(name="n")
            fig = px.bar(grp, x=grp.columns[0], y="n",
                         color=tile["group_col"], barmode="stack", title=title)
        else:
            fig = px.bar(x=counts.index.astype(str), y=counts.values,
                         title=title)

    elif chart == "Line chart":
        fig = px.line(x=counts.index.astype(str), y=counts.values,
                      markers=True, title=title,
                      labels={"x": "", "y": "Count"})

    elif chart == "Mean line":
        vals = pd.to_numeric(series, errors="coerce").dropna()
        mean_val = vals.mean()
        fig = px.bar(x=counts.index.astype(str), y=counts.values,
                     title=f"{title}  (mean={mean_val:.2f})")
        fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.2f}")

    elif chart == "Heatmap":
        fig = px.imshow(
            pd.crosstab(series, df.iloc[:, 0]) if len(df.columns) > 1 else counts.to_frame(),
            title=title, color_continuous_scale="Blues"
        )

    elif chart == "Grouped bar":
        fig = px.bar(x=counts.index.astype(str), y=counts.values, title=title)

    elif chart == "Word count bar":
        words = series.dropna().astype(str).str.lower().str.split().explode()
        wc = words.value_counts().head(20)
        fig = px.bar(x=wc.values, y=wc.index, orientation="h",
                     title=f"{title} — top words")
        fig.update_layout(yaxis=dict(autorange="reversed"))

    else:
        fig = px.bar(x=counts.index.astype(str), y=counts.values, title=title)

    fig.update_layout(
        height=300, margin=dict(t=40, b=20, l=20, r=20),
        font=dict(size=11),
        title_font_size=13,
    )
    return fig


# ── PDF export ────────────────────────────────────────────────────────────────

def _export_pdf(tiles, df):
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4),
                            leftMargin=1*cm, rightMargin=1*cm,
                            topMargin=1*cm, bottomMargin=1*cm)
    story = []

    for tile in tiles:
        col = tile.get("dataset_col")
        if not col or col not in df.columns:
            continue
        fig = _render_chart(df[col], tile, df)
        img_buf = io.BytesIO(fig.to_image(format="png", width=700, height=350))
        story.append(RLImage(img_buf, width=18*cm, height=9*cm))
        story.append(Paragraph(
            f"<b>{tile['code']}</b> — {tile['question'][:120]}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5*cm))

    doc.build(story)
    buf.seek(0)
    return buf


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("2b_viz", "Parse questionnaire, map variables, build interactive dashboard.")

    # ── 1. Questionnaire upload & parse ───────────────────────────────────────
    st.markdown("### Step 1 — Upload questionnaire document")
    qnr_file = st.file_uploader(
        "Upload questionnaire (Word, PDF, Excel, TXT)",
        type=["docx", "pdf", "xlsx", "xls", "txt"],
        key="qnr_upload"
    )

    if qnr_file:
        suffix = Path(qnr_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(qnr_file.read())
            tmp_path = tmp.name

        with st.spinner("Parsing questionnaire..."):
            parsed = parse_questionnaire(tmp_path)
        os.unlink(tmp_path)

        st.success(f"Parsed **{len(parsed)} questions** from `{qnr_file.name}`")
        st.session_state["parsed_questions"] = parsed

    # Allow re-use of previously parsed questions
    if not st.session_state.get("parsed_questions"):
        st.info("Upload a questionnaire to auto-fill the codebook.")
        parsed = []
    else:
        parsed = st.session_state["parsed_questions"]

    st.divider()

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    st.markdown("### Step 2 — Dataset")

    df = None
    if st.session_state.get("X_train") is not None:
        target = st.session_state.get("target_col", "")
        X = st.session_state["X_train"]
        y = st.session_state.get("y_train")
        df = X.copy()
        if y is not None:
            df[target] = y.values
        st.success(f"Using dataset from Stage 0/1 — {df.shape[0]} rows, {df.shape[1]} columns.")
    else:
        up = st.file_uploader("Or upload dataset directly (CSV/Excel)", type=["csv","xlsx","xls"],
                               key="viz_data_upload")
        if up:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    if df is None:
        st.warning("No dataset loaded. Complete Stage 0 or upload a file above.")
        return

    st.divider()

    # ── 3. Build / edit codebook ──────────────────────────────────────────────
    st.markdown("### Step 3 — Codebook editor")
    st.caption("Auto-parsed from questionnaire. Fix any mistakes — add questions, reassign options and variables manually.")

    # ── Initialise codebook ───────────────────────────────────────────────────
    if "codebook" not in st.session_state or st.session_state["codebook"] is None:
        if parsed:
            with st.spinner("Auto-mapping dataset columns to questionnaire..."):
                codebook = build_codebook(df, parsed)
            n_mapped = sum(1 for t in codebook if t["include"] and t["dataset_col"])
            st.success(f"Auto-mapped **{n_mapped}** questions to dataset columns.")
        else:
            col_groups = group_columns(df)
            codebook = []
            for prefix, grp in col_groups.items():
                vt = "numeric" if pd.api.types.is_numeric_dtype(df[grp["primary"]]) \
                     else "categorical"
                codebook.append({
                    "code": prefix, "question": prefix,
                    "var_type": vt,
                    "chart_type": CHART_OPTIONS.get(vt, ["Bar chart"])[0],
                    "group_type": grp["group_type"],
                    "dataset_col": grp["primary"],
                    "all_cols": grp["cols"],
                    "options": [],
                    "value_labels": {}, "scale_low": "", "scale_high": "",
                    "scale_points": None, "pn_notes": "", "include": True,
                })
        # Ensure every tile has an "options" list
        for t in codebook:
            if "options" not in t:
                t["options"] = []
        st.session_state["codebook"] = codebook

    codebook  = st.session_state["codebook"]
    all_cols  = ["(not mapped)"] + list(df.columns)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    tb1, tb2, tb3 = st.columns([2, 2, 3])
    search_term = tb1.text_input("Search questions", placeholder="type to filter…", key="cb_search")
    show_unmapped = tb2.checkbox("Show unmapped only", False, key="cb_unmapped")
    if tb3.button("+ Add new question", key="add_q_btn"):
        st.session_state["show_add_form"] = True

    if st.session_state.get("show_add_form"):
        with st.form("add_question_form"):
            fa, fb, fc, fd = st.columns([1, 3, 2, 1])
            new_code = fa.text_input("Code*")
            new_q    = fb.text_input("Question text*")
            new_col  = fc.selectbox("Primary dataset column", all_cols)
            new_type = fd.selectbox("Type", VAR_TYPES)
            submitted = st.form_submit_button("Add question")
            if submitted and new_code and new_q:
                codebook.append({
                    "code": new_code.upper(), "question": new_q,
                    "var_type": new_type,
                    "chart_type": CHART_OPTIONS.get(new_type, ["Bar chart"])[0],
                    "group_type": "single",
                    "dataset_col": new_col if new_col != "(not mapped)" else "",
                    "all_cols": [new_col] if new_col != "(not mapped)" else [],
                    "options": [],
                    "value_labels": {}, "scale_low": "", "scale_high": "",
                    "scale_points": None, "pn_notes": "", "include": True,
                })
                st.session_state["codebook"] = codebook
                st.session_state["show_add_form"] = False
                st.rerun()

    st.divider()

    # ── Per-question editor ───────────────────────────────────────────────────
    # Classify each tile
    def _is_qnr_matched(tile):
        """True if tile came from a parsed QNR question (matched to a dataset column)."""
        return tile.get("include", False) and bool(tile.get("dataset_col"))

    def _is_unmatched(tile):
        """True if tile is a dataset column that could not be matched to any QNR question."""
        return not tile.get("include", True) or not tile.get("dataset_col")

    matched_count   = sum(1 for t in codebook if _is_qnr_matched(t))
    unmatched_count = sum(1 for t in codebook if _is_unmatched(t))

    # Toolbar summary counts
    st.markdown(
        f"<span style='color:#2E75B6;font-weight:bold'>✅ {matched_count} matched</span> &nbsp;|&nbsp; "
        f"<span style='color:#C0504D;font-weight:bold'>⚠ {unmatched_count} unmatched / unmapped</span> &nbsp;|&nbsp; "
        f"Total: {len(codebook)}",
        unsafe_allow_html=True
    )

    # Filter
    def _passes_filter(tile):
        code_match = (not search_term or
                      search_term.lower() in tile["code"].lower() or
                      search_term.lower() in tile["question"].lower())
        unmap_match = (not show_unmapped or _is_unmatched(tile))
        return code_match and unmap_match

    visible = [(idx, t) for idx, t in enumerate(codebook) if _passes_filter(t)]

    # Split into two groups for display
    matched_visible   = [(idx, t) for idx, t in visible if _is_qnr_matched(t)]
    unmatched_visible = [(idx, t) for idx, t in visible if _is_unmatched(t)]

    def _render_tile_editor(idx, tile):
        """Render the interior of a single question expander."""
        # ── Row 1: include / code / question text ──────────────────────────
        rc1, rc2, rc3 = st.columns([1, 1, 4])
        tile["include"]  = rc1.checkbox("Include in dashboard",
                                        tile["include"], key=f"inc_{idx}")
        tile["code"]     = rc2.text_input("Q Code", tile["code"],
                                          key=f"code_{idx}")
        tile["question"] = rc3.text_area("Question text", tile["question"],
                                         height=70, key=f"q_{idx}")

        # ── Row 2: type / chart / primary column ───────────────────────────
        rt1, rt2, rt3 = st.columns(3)
        vt_idx = VAR_TYPES.index(tile["var_type"]) \
                 if tile["var_type"] in VAR_TYPES else 0
        tile["var_type"] = rt1.selectbox("Variable type", VAR_TYPES,
                                         index=vt_idx, key=f"vt_{idx}")

        chart_opts = CHART_OPTIONS.get(tile["var_type"], ["Bar chart"])
        ct_idx = chart_opts.index(tile["chart_type"]) \
                 if tile["chart_type"] in chart_opts else 0
        tile["chart_type"] = rt2.selectbox("Chart type", chart_opts,
                                           index=ct_idx, key=f"ct_{idx}")

        curr_col = tile.get("dataset_col", "")
        ci = all_cols.index(curr_col) if curr_col in all_cols else 0
        tile["dataset_col"] = rt3.selectbox("Primary column", all_cols,
                                            index=ci, key=f"pcol_{idx}")

        # ── Scale anchors ──────────────────────────────────────────────────
        if tile["var_type"] in ("scale_7", "scale_5", "ordinal"):
            sa1, sa2 = st.columns(2)
            tile["scale_low"]  = sa1.text_input("Scale low label",
                                                tile.get("scale_low", ""),
                                                key=f"sl_{idx}")
            tile["scale_high"] = sa2.text_input("Scale high label",
                                                tile.get("scale_high", ""),
                                                key=f"sh_{idx}")

        st.markdown("**Option / Row mapping** — each row is one answer option or grid row")
        st.caption("Edit directly. + button adds rows. Assign dataset columns per option.")

        # ── Build options dataframe ────────────────────────────────────────
        existing_opts = tile.get("options", [])
        if not existing_opts and tile.get("all_cols"):
            for c in tile["all_cols"]:
                existing_opts.append({
                    "option_label": c,
                    "dataset_col":  c if c in df.columns else "",
                    "value_coding": "",
                    "notes":        "",
                })

        opts_df = pd.DataFrame(
            existing_opts if existing_opts else
            [{"option_label": "", "dataset_col": "", "value_coding": "", "notes": ""}],
            columns=["option_label", "dataset_col", "value_coding", "notes"]
        )

        edited = st.data_editor(
            opts_df,
            key=f"opts_{idx}",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "option_label": st.column_config.TextColumn(
                    "Option / Row label", width="medium"),
                "dataset_col": st.column_config.SelectboxColumn(
                    "Dataset column", options=all_cols, width="medium"),
                "value_coding": st.column_config.TextColumn(
                    "Value coding",
                    help="e.g. '1=Aware, 2=Not Aware'", width="medium"),
                "notes": st.column_config.TextColumn("Notes", width="small"),
            },
        )

        tile["options"] = edited.to_dict("records")

        # Sync all_cols from options table
        mapped_cols = [r["dataset_col"] for r in tile["options"]
                       if r.get("dataset_col") and r["dataset_col"] in df.columns]
        if mapped_cols:
            tile["all_cols"] = mapped_cols
            if not tile.get("dataset_col") or tile["dataset_col"] not in df.columns:
                tile["dataset_col"] = mapped_cols[0]

        # ── Value labels ───────────────────────────────────────────────────
        if tile["var_type"] in ("categorical", "ordinal", "single") \
                and tile.get("dataset_col") and tile["dataset_col"] in df.columns:
            uvals = df[tile["dataset_col"]].dropna().unique().tolist()
            if len(uvals) <= 20:
                st.caption("Quick value labels:")
                lc = st.columns(min(4, len(uvals)))
                for vi, v in enumerate(uvals):
                    cur = tile["value_labels"].get(str(v), str(v))
                    tile["value_labels"][str(v)] = lc[vi % 4].text_input(
                        f"'{v}'", cur, key=f"lbl_{idx}_{vi}")

        # ── Delete ─────────────────────────────────────────────────────────
        if st.button("Delete this question", key=f"del_{idx}", type="secondary"):
            codebook.pop(idx)
            st.session_state["codebook"] = codebook
            st.rerun()

    # ── Matched questions (from QNR) ──────────────────────────────────────────
    if matched_visible:
        st.markdown(
            "<div style='background:#e8f4e8;padding:6px 10px;border-radius:4px;"
            "font-weight:bold;color:#2e6b2e'>✅ Matched questions — parsed from QNR and "
            "mapped to dataset columns</div>",
            unsafe_allow_html=True
        )
        for idx, tile in matched_visible:
            with st.expander(
                f"✅ **{tile['code']}**  —  {tile['question'][:80]}",
                expanded=False
            ):
                _render_tile_editor(idx, tile)

    # ── Unmatched columns (dataset-only, no QNR match) ────────────────────────
    if unmatched_visible:
        st.markdown(
            "<div style='background:#fff3cd;padding:6px 10px;border-radius:4px;"
            "font-weight:bold;color:#7d5a00;margin-top:12px'>⚠ Unmatched columns — "
            "found in dataset but could not be assigned to any QNR question. "
            "Assign a question or exclude them below.</div>",
            unsafe_allow_html=True
        )
        # Bulk-exclude button
        if st.button(f"Exclude all {len(unmatched_visible)} unmatched from dashboard",
                     key="exclude_all_unmatched"):
            for idx, tile in unmatched_visible:
                codebook[idx]["include"] = False
            st.session_state["codebook"] = codebook
            st.rerun()

        for idx, tile in unmatched_visible:
            label = f"⚠ **{tile['code']}**  —  {tile['question'][:80]}"
            with st.expander(label, expanded=False):
                st.caption(
                    "This column has no matching QNR question. "
                    "You can assign it to a question below, include it as-is, or exclude it."
                )
                _render_tile_editor(idx, tile)

    if not matched_visible and not unmatched_visible:
        st.info("No questions match the current filter.")

    st.session_state["codebook"] = codebook
    mapped = [t for t in codebook
              if t.get("include") and t.get("dataset_col")
              and t["dataset_col"] in df.columns]
    st.caption(f"**{len(mapped)} / {len(codebook)}** questions mapped and included in dashboard.")

    st.divider()

    # ── 4. Dashboard layout controls ─────────────────────────────────────────
    st.markdown("### Step 4 — Dashboard")
    if not mapped:
        st.warning("Map at least one column to a question above to see charts.")
        return

    lc1, lc2, lc3, lc4 = st.columns(4)
    mode        = lc1.radio("Layout mode", ["Grid", "Slide"], horizontal=True)
    tiles_per_row = lc2.select_slider("Tiles per row", [1, 2, 3, 4], value=3) \
                   if mode == "Grid" else 1
    tiles_per_slide = lc3.select_slider("Tiles per slide", list(range(1, 13)), value=6) \
                      if mode == "Slide" else len(mapped)
    show_pn     = lc4.checkbox("Show PN notes", False)

    # Slide navigation state
    if "viz_slide" not in st.session_state:
        st.session_state["viz_slide"] = 0

    total_slides = max(1, -(-len(mapped) // tiles_per_slide))  # ceiling div

    if mode == "Slide":
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        if nav1.button("◀ Prev") and st.session_state["viz_slide"] > 0:
            st.session_state["viz_slide"] -= 1
        if nav3.button("Next ▶") and st.session_state["viz_slide"] < total_slides - 1:
            st.session_state["viz_slide"] += 1
        nav2.markdown(
            f"<div style='text-align:center;padding-top:8px'>"
            f"Slide {st.session_state['viz_slide']+1} / {total_slides}</div>",
            unsafe_allow_html=True
        )
        slide_start = st.session_state["viz_slide"] * tiles_per_slide
        tiles_to_show = mapped[slide_start : slide_start + tiles_per_slide]
    else:
        tiles_to_show = mapped

    # ── Render tiles ──────────────────────────────────────────────────────────
    rows = [tiles_to_show[i:i+tiles_per_row]
            for i in range(0, len(tiles_to_show), tiles_per_row)]

    for row_tiles in rows:
        ui_cols = st.columns(len(row_tiles))
        for ui_col, tile in zip(ui_cols, row_tiles):
            with ui_col:
                col = tile["dataset_col"]
                st.markdown(
                    f"**{tile['code']}** — "
                    f"<span style='font-size:12px;color:gray'>{tile['question'][:60]}{'…' if len(tile['question'])>60 else ''}</span>",
                    unsafe_allow_html=True
                )
                if tile.get("scale_low") and tile.get("scale_high"):
                    st.caption(f"1={tile['scale_low']} → {tile['scale_points'] if tile.get('scale_points') else ''}={tile['scale_high']}")

                try:
                    g_type = tile.get("group_type", "single")
                    if g_type == "grid":
                        fig = _render_grid_chart(df, tile)
                    elif g_type == "multi" and len(tile.get("all_cols", [])) > 1:
                        fig = _render_multi_chart(df, tile)
                    else:
                        fig = _render_chart(df[col], tile, df)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"chart_{tile['code']}")
                except Exception as e:
                    st.error(f"Chart error: {e}")

                if show_pn and tile.get("pn_notes"):
                    st.caption(f"PN: {tile['pn_notes'][:100]}")

    st.divider()

    # ── 5. PDF export ─────────────────────────────────────────────────────────
    st.markdown("### Step 5 — Export")
    if st.button("Export all charts as PDF"):
        try:
            with st.spinner("Generating PDF..."):
                pdf_buf = _export_pdf(mapped, df)
            st.download_button(
                "Download PDF",
                data=pdf_buf,
                file_name="viz_dashboard.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF export error: {e}. Try installing: pip install kaleido")
            st.info("Tip: run  pip install kaleido  then restart the app.")
