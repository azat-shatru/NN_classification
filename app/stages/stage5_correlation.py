"""
Stage 5 — Correlation Filter
Remove highly correlated features to reduce redundancy before modelling.

Steps:
  1. Compute Pearson correlation matrix on scaled X_train
  2. Visualise as interactive heatmap (Plotly)
  3. Flag pairs above threshold
  4. User selects which columns to drop (with context)
  5. Apply removal to X_train + X_test
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.state import init
from utils.ui import stage_header


def _corr_pairs(corr_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return DataFrame of column pairs with |correlation| >= threshold."""
    rows = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                rows.append({
                    "Column A": cols[i],
                    "Column B": cols[j],
                    "Correlation": round(val, 4),
                    "|r|": round(abs(val), 4),
                })
    return pd.DataFrame(rows).sort_values("|r|", ascending=False).reset_index(drop=True)


def show():
    init()
    stage_header("5_correlation", "Identify and remove highly correlated features.")

    if not st.session_state.get("df_scaled"):
        st.warning("Complete **Stage 4 · Scaling** first.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()

    numeric_cols = [c for c in X_train.columns
                    if pd.api.types.is_numeric_dtype(X_train[c])]

    st.markdown(f"Working with **{len(numeric_cols)} numeric columns** on the scaled train set.")

    st.divider()

    # ── Step 1: Compute correlation ───────────────────────────────────────────
    st.markdown("### Step 1 — Correlation heatmap")
    threshold = st.slider(
        "Highlight pairs with |r| ≥",
        0.5, 1.0, 0.8, 0.05,
        key="corr_thresh",
        help="Pearson |r| ≥ 0.9 is a typical drop threshold; 0.7–0.8 is moderate."
    )

    if len(numeric_cols) > 80:
        st.info(
            f"Large feature set ({len(numeric_cols)} cols). "
            "Showing top-50 by variance for the heatmap."
        )
        var_order = X_train[numeric_cols].var().sort_values(ascending=False)
        plot_cols = var_order.head(50).index.tolist()
    else:
        plot_cols = numeric_cols

    corr = X_train[plot_cols].corr()

    # Plotly heatmap
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Pearson Correlation Matrix (scaled train set)",
    )
    fig.update_layout(
        height=max(500, min(len(plot_cols) * 14, 900)),
        margin=dict(t=50, b=20, l=20, r=20),
        coloraxis_colorbar=dict(title="|r|"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Step 2: Flagged pairs ─────────────────────────────────────────────────
    st.markdown("### Step 2 — Highly correlated pairs")

    full_corr = X_train[numeric_cols].corr()
    pairs_df  = _corr_pairs(full_corr, threshold)

    if pairs_df.empty:
        st.success(f"No column pairs exceed |r| = {threshold}. No filtering needed.")
    else:
        st.warning(f"**{len(pairs_df)} pairs** exceed |r| = {threshold}.")
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)

        # Suggest columns to drop (the second column in each redundant pair)
        suggested_drops = []
        dropped_set = set()
        for _, row in pairs_df.iterrows():
            a, b = row["Column A"], row["Column B"]
            if a not in dropped_set and b not in dropped_set:
                suggested_drops.append(b)
                dropped_set.add(b)

        st.divider()

        # ── Step 3: Column-level deep-dive ────────────────────────────────────
        st.markdown("### Step 3 — Inspect a column pair")
        if not pairs_df.empty:
            pair_labels = [f"{r['Column A']}  ↔  {r['Column B']}  (r={r['Correlation']})"
                           for _, r in pairs_df.iterrows()]
            chosen_pair = st.selectbox("Select pair to inspect", pair_labels, key="corr_pair_sel")
            idx = pair_labels.index(chosen_pair)
            col_a = pairs_df.iloc[idx]["Column A"]
            col_b = pairs_df.iloc[idx]["Column B"]

            p1, p2 = st.columns(2)
            with p1:
                fig2 = px.scatter(
                    x=X_train[col_a], y=X_train[col_b],
                    labels={"x": col_a, "y": col_b},
                    title=f"Scatter: {col_a} vs {col_b}",
                    opacity=0.4,
                )
                fig2.update_layout(height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)
            with p2:
                st.markdown(f"**{col_a}** stats")
                st.dataframe(X_train[col_a].describe().to_frame().T.round(3),
                             use_container_width=True)
                st.markdown(f"**{col_b}** stats")
                st.dataframe(X_train[col_b].describe().to_frame().T.round(3),
                             use_container_width=True)

        st.divider()

        # ── Step 4: Select columns to drop ────────────────────────────────────
        st.markdown("### Step 4 — Select columns to drop")
        st.caption(
            "Default suggestion: drop the second column in each correlated pair "
            "(keeps the first, which was listed as Column A). "
            "Override freely — use domain knowledge."
        )

        cols_to_drop = st.multiselect(
            "Columns to remove from feature set:",
            options=numeric_cols,
            default=suggested_drops,
            key="corr_drop_sel",
        )

        st.info(f"Dropping **{len(cols_to_drop)}** columns — "
                f"**{len(numeric_cols) - len(cols_to_drop)}** will remain.")

        st.divider()

        # ── Step 5: Apply ──────────────────────────────────────────────────────
        st.markdown("### Step 5 — Apply filter")
        if st.button("Remove selected columns", type="primary", key="apply_corr"):
            X_tr_new = X_train.drop(columns=cols_to_drop, errors="ignore")
            X_te_new = X_test.drop(columns=cols_to_drop, errors="ignore")
            st.session_state["X_train"]   = X_tr_new
            st.session_state["X_test"]    = X_te_new
            st.session_state["corr_done"] = True
            st.session_state["corr_dropped_cols"] = cols_to_drop

            st.success(
                f"Removed {len(cols_to_drop)} correlated columns. "
                f"Remaining features: **{X_tr_new.shape[1]}**"
            )
            st.rerun()

    # ── Skip / mark done ──────────────────────────────────────────────────────
    if not st.session_state.get("corr_done"):
        if pairs_df.empty or st.button("Skip — no columns to drop", key="corr_skip"):
            st.session_state["corr_done"] = True
            st.success("Correlation filter complete. Proceed to Stage 6 · RF Importance.")
            st.rerun()
    else:
        dropped = st.session_state.get("corr_dropped_cols", [])
        X_tr = st.session_state["X_train"]
        st.info(
            f"Correlation filter done — "
            f"{'dropped ' + str(len(dropped)) + ' columns, ' if dropped else ''}"
            f"**{X_tr.shape[1]} features** remain."
        )
        if st.button("Redo correlation filter", key="redo_corr"):
            st.session_state["corr_done"] = False
            st.session_state["corr_dropped_cols"] = []
            st.rerun()
