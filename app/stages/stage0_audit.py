"""
Stage 0 — Data Audit
Uploads the dataset, performs a visual quality audit, and stores the
dataframe in session state for all downstream stages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
import matplotlib.pyplot as plt
import io

from utils.state import init
from utils.ui import stage_header


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_file(uploaded) -> pd.DataFrame:
    if uploaded.name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def _missing_heatmap(df: pd.DataFrame):
    """Return a missingno matrix as a PNG bytes object."""
    fig, ax = plt.subplots(figsize=(min(18, max(8, len(df.columns) * 0.35)), 4))
    msno.matrix(df, ax=ax, sparkline=False, fontsize=9, color=(0.18, 0.46, 0.71))
    ax.set_title("Missing-value pattern (white = missing)", fontsize=11, pad=8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _skewness_label(s: float) -> str:
    if abs(s) < 0.5:   return "✅ normal"
    if abs(s) < 1.0:   return "⚠️ moderate"
    return "🔴 high"


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("0_audit", "Upload your dataset and review its quality before any processing.")

    # ── 1. File upload ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Waiting for a file…")
        return

    df = _load_file(uploaded)
    st.session_state["raw_df"] = df.copy()
    st.session_state["df"]     = df.copy()

    st.success(f"Loaded **{df.shape[0]} rows × {df.shape[1]} columns**")

    # ── 2. Target column selection ────────────────────────────────────────────
    st.markdown("### Select target (label) column")
    target = st.selectbox("Target column", df.columns.tolist(),
                          index=len(df.columns) - 1)
    st.session_state["target_col"] = target

    # ── 3. Column editor — all properties + type + drop in one sortable table ──
    st.markdown("### Column editor")
    st.caption(
        "Every column is one row. **Sort** by clicking any header. "
        "Change **Type** inline. Tick **Drop** to remove before processing. "
        "Click **Apply changes** when done."
    )

    feature_cols = [c for c in df.columns if c != target]
    auto_numeric = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Build the editor dataframe (one row per feature column)
    # Preserve prior edits if they exist in session state
    prior = st.session_state.get("col_editor_state", {})

    editor_rows = []
    for col in feature_cols:
        s        = df[col]
        is_num   = pd.api.types.is_numeric_dtype(s)
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

        editor_rows.append({
            "Column":    col,
            "Type":      default_type,
            "Drop":      default_drop,
            "Missing %": miss_pct,
            "Zero %":    zero_pct,
            "Variance":  variance,
            "Skewness":  skew_val,
            "Unique":    n_unique,
            "Dtype":     str(s.dtype),
        })

    editor_df = pd.DataFrame(editor_rows)

    edited = st.data_editor(
        editor_df,
        key="col_editor_table",
        use_container_width=True,
        height=min(600, max(300, len(editor_rows) * 36 + 40)),
        hide_index=True,
        column_config={
            "Column":    st.column_config.TextColumn("Column", disabled=True, width="medium"),
            "Type":      st.column_config.SelectboxColumn(
                             "Type", options=["numeric", "categorical", "ordinal"],
                             width="small", required=True),
            "Drop":      st.column_config.CheckboxColumn("Drop", width="small"),
            "Missing %": st.column_config.NumberColumn("Missing %", format="%.1f",
                             disabled=True, width="small"),
            "Zero %":    st.column_config.NumberColumn("Zero %", format="%.1f",
                             disabled=True, width="small"),
            "Variance":  st.column_config.NumberColumn("Variance", format="%.4f",
                             disabled=True, width="small"),
            "Skewness":  st.column_config.NumberColumn("Skewness", format="%.3f",
                             disabled=True, width="small"),
            "Unique":    st.column_config.NumberColumn("Unique", disabled=True, width="small"),
            "Dtype":     st.column_config.TextColumn("Dtype", disabled=True, width="small"),
        },
    )

    # ── Live summary under the table ──────────────────────────────────────────
    n_drop_live  = int(edited["Drop"].sum())
    n_num_live   = int((edited.loc[~edited["Drop"], "Type"] == "numeric").sum())
    n_cat_live   = int((edited.loc[~edited["Drop"], "Type"] == "categorical").sum())
    n_ord_live   = int((edited.loc[~edited["Drop"], "Type"] == "ordinal").sum())

    lv1, lv2, lv3, lv4 = st.columns(4)
    lv1.metric("Queued to drop", n_drop_live,
               delta=f"−{n_drop_live}" if n_drop_live else None, delta_color="inverse")
    lv2.metric("Numeric (kept)",      n_num_live)
    lv3.metric("Categorical (kept)",  n_cat_live)
    lv4.metric("Ordinal (kept)",      n_ord_live)

    if n_drop_live:
        drop_names = edited.loc[edited["Drop"], "Column"].tolist()
        st.warning(f"Marked for removal: {', '.join(drop_names[:15])}"
                   f"{'…' if len(drop_names) > 15 else ''}")

    if st.button("Apply changes", type="primary", key="apply_col_editor"):
        # Persist editor choices to session state
        st.session_state["col_editor_state"] = {
            row["Column"]: {"Type": row["Type"], "Drop": row["Drop"]}
            for _, row in edited.iterrows()
        }

        kept  = edited[~edited["Drop"]]
        drops = edited[edited["Drop"]]["Column"].tolist()

        st.session_state["numeric_cols"]     = kept.loc[kept["Type"]=="numeric",     "Column"].tolist()
        st.session_state["categorical_cols"] = kept.loc[kept["Type"]=="categorical", "Column"].tolist()
        st.session_state["ordinal_cols"]     = kept.loc[kept["Type"]=="ordinal",     "Column"].tolist()
        st.session_state["dropped_cols_audit"] = drops

        # Apply drops to df immediately so downstream sees the clean set
        clean_df = df.drop(columns=drops, errors="ignore")
        st.session_state["df"] = clean_df

        st.success(
            f"Applied — keeping **{len(kept)} columns** "
            f"({n_num_live} numeric, {n_cat_live} categorical, {n_ord_live} ordinal). "
            f"Dropped **{len(drops)}**."
        )
        st.rerun()

    else:
        # Read back from prior state if already applied once
        if prior:
            kept  = editor_df[~editor_df["Drop"]]
            st.session_state["numeric_cols"]     = kept.loc[kept["Type"]=="numeric",     "Column"].tolist()
            st.session_state["categorical_cols"] = kept.loc[kept["Type"]=="categorical", "Column"].tolist()
            st.session_state["ordinal_cols"]     = kept.loc[kept["Type"]=="ordinal",     "Column"].tolist()
        else:
            st.session_state["numeric_cols"]     = [c for c in feature_cols if c in auto_numeric]
            st.session_state["categorical_cols"] = [c for c in feature_cols if c not in auto_numeric]
            st.session_state["ordinal_cols"]     = []

    st.divider()

    # ── 4. Basic shape & type summary ─────────────────────────────────────────
    st.markdown("### Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",        df.shape[0])
    c2.metric("Columns",     df.shape[1])
    c3.metric("Numeric",     len(st.session_state["numeric_cols"]))
    c4.metric("Categorical / Ordinal",
              len(st.session_state["categorical_cols"]) +
              len(st.session_state["ordinal_cols"]))

    # ── 5. Missing values ─────────────────────────────────────────────────────
    st.markdown("### Missing values")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({
        "Column": miss.index,
        "Missing count": miss.values,
        "Missing %": miss_pct.values
    }).query("`Missing count` > 0").sort_values("Missing %", ascending=False)

    if miss_df.empty:
        st.success("No missing values found.")
    else:
        st.warning(f"{len(miss_df)} columns have missing values.")
        tab1, tab2, tab3 = st.tabs(["Table", "Bar chart", "Pattern matrix"])

        with tab1:
            st.dataframe(miss_df.reset_index(drop=True), use_container_width=True)

        with tab2:
            fig = px.bar(miss_df, x="Column", y="Missing %",
                         color="Missing %",
                         color_continuous_scale="Reds",
                         title="Missing value % per column")
            fig.add_hline(y=40, line_dash="dash", line_color="red",
                          annotation_text="40% threshold")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.image(_missing_heatmap(df), use_container_width=True)

    # ── 6. Duplicate rows ─────────────────────────────────────────────────────
    st.markdown("### Duplicate rows")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        st.success("No duplicate rows.")
    else:
        st.warning(f"{n_dup} duplicate rows found.")
        if st.checkbox("Preview duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].head(20), use_container_width=True)

    # ── 7. Class label distribution ───────────────────────────────────────────
    st.markdown(f"### Target distribution — `{target}`")
    class_counts = df[target].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(df) * 100).round(1)

    fig = px.bar(class_counts, x="Class", y="Count",
                 text="Percentage",
                 color="Class",
                 title=f"Class distribution for '{target}'")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    max_pct = class_counts["Percentage"].max()
    min_pct = class_counts["Percentage"].min()
    imbalance = max_pct - min_pct
    if imbalance <= 20:
        st.success(f"Classes are balanced (max spread: {imbalance:.1f}%)")
    else:
        st.warning(f"Class imbalance detected (max spread: {imbalance:.1f}%)")

    # ── 8. Numeric statistics ─────────────────────────────────────────────────
    num_cols = st.session_state["numeric_cols"]
    if num_cols:
        st.markdown("### Numeric column statistics")
        stats = df[num_cols].describe().T
        stats["skewness"]  = df[num_cols].skew().round(3)
        stats["kurtosis"]  = df[num_cols].kurtosis().round(3)
        stats["skew_flag"] = stats["skewness"].apply(_skewness_label)
        st.dataframe(stats.style.background_gradient(subset=["skewness"], cmap="RdYlGn_r"),
                     use_container_width=True)

    # ── 9. Categorical cardinality ────────────────────────────────────────────
    cat_cols = st.session_state["categorical_cols"] + st.session_state["ordinal_cols"]
    if cat_cols:
        st.markdown("### Categorical / Ordinal — unique value counts")
        card = pd.DataFrame({
            "Column":        cat_cols,
            "Unique values": [df[c].nunique() for c in cat_cols],
            "Type":          ["categorical" if c in st.session_state["categorical_cols"]
                              else "ordinal" for c in cat_cols]
        }).sort_values("Unique values", ascending=False)

        fig = px.bar(card, x="Column", y="Unique values", color="Type",
                     title="Cardinality per categorical/ordinal column",
                     color_discrete_map={"categorical": "#2E75B6", "ordinal": "#ED7D31"})
        st.plotly_chart(fig, use_container_width=True)

    # ── 10. Quick-flag bulk-tick (feeds the Drop column in the editor above) ────
    st.divider()
    st.markdown("### Quick-flag sweep — pre-tick Drop in the editor above")
    st.caption(
        "These buttons mark matching columns as **Drop=✓** in the table above. "
        "Then click **Apply changes** to lock your selection."
    )

    feature_cols_all = [c for c in df.columns if c != target]

    # Compute flags
    def _col_stats(col):
        s = df[col]
        n_miss = s.isnull().sum()
        non_null = s.dropna()
        is_num = pd.api.types.is_numeric_dtype(s)
        num_nn = pd.to_numeric(non_null, errors="coerce").dropna() if is_num else pd.Series([], dtype=float)
        variance  = float(num_nn.var()) if len(num_nn) > 1 else None
        zero_pct  = float((num_nn == 0).sum() / max(len(num_nn), 1) * 100) if len(num_nn) > 0 else 0.0
        return {
            "all_null":  n_miss == len(s),
            "all_zero":  is_num and len(num_nn) > 0 and float(num_nn.abs().max()) == 0,
            "zero_var":  variance is not None and variance == 0.0,
            "miss_pct":  round(n_miss / max(len(s), 1) * 100, 1),
            "variance":  variance,
            "zero_pct":  zero_pct,
        }

    col_stats = {c: _col_stats(c) for c in feature_cols_all}

    sw1, sw2, sw3 = st.columns(3)
    miss_thresh     = sw1.slider("Missing % above",   0, 100,  50, 5,   key="sw_miss")
    var_thresh      = sw2.slider("Variance below",    0.0, 1.0, 0.01, 0.001,
                                 format="%.3f",       key="sw_var")
    zero_pct_thresh = sw3.slider("Zero % above",      0, 100,  90, 5,   key="sw_zero")

    flagged_null  = [c for c, s in col_stats.items() if s["all_null"]]
    flagged_zero  = [c for c, s in col_stats.items() if s["all_zero"] or s["zero_var"]]
    flagged_var   = [c for c, s in col_stats.items()
                     if s["variance"] is not None and s["variance"] < var_thresh]
    flagged_miss  = [c for c, s in col_stats.items() if s["miss_pct"] >= miss_thresh]
    flagged_zp    = [c for c, s in col_stats.items() if s["zero_pct"] >= zero_pct_thresh]
    flagged_all   = list(set(flagged_null + flagged_zero + flagged_var +
                             flagged_miss + flagged_zp))

    # Summary counts
    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fm1.metric("All-null",          len(flagged_null))
    fm2.metric("All-zero/const",    len(flagged_zero))
    fm3.metric(f"Low var <{var_thresh}", len(flagged_var))
    fm4.metric(f"Missing >{miss_thresh}%", len(flagged_miss))
    fm5.metric(f"Zero% >{zero_pct_thresh}%", len(flagged_zp))

    def _tick_drop(cols_to_flag):
        """Mark these columns as Drop=True in col_editor_state."""
        state = dict(st.session_state.get("col_editor_state", {}))
        for col in feature_cols_all:
            entry = state.get(col, {
                "Type": "numeric" if col in auto_numeric else "categorical",
                "Drop": False
            })
            if col in cols_to_flag:
                entry["Drop"] = True
            state[col] = entry
        st.session_state["col_editor_state"] = state

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button(f"Tick all-null ({len(flagged_null)})",
                 key="sw_btn_null", disabled=not flagged_null):
        _tick_drop(flagged_null); st.rerun()

    if b2.button(f"Tick all-zero ({len(flagged_zero)})",
                 key="sw_btn_zero", disabled=not flagged_zero):
        _tick_drop(flagged_zero); st.rerun()

    if b3.button(f"Tick low-var ({len(flagged_var)})",
                 key="sw_btn_var", disabled=not flagged_var):
        _tick_drop(flagged_var); st.rerun()

    if b4.button(f"Tick ALL flagged ({len(flagged_all)})",
                 key="sw_btn_all", type="primary", disabled=not flagged_all):
        _tick_drop(flagged_all); st.rerun()

    if b5.button("Clear all Drop ticks", key="sw_btn_clear"):
        _tick_drop([])          # tick none — also un-ticks by rebuilding state
        state = dict(st.session_state.get("col_editor_state", {}))
        for col in state:
            state[col]["Drop"] = False
        st.session_state["col_editor_state"] = state
        st.rerun()

    st.info("After ticking, scroll up to the **Column editor** table and click **Apply changes**.")
