"""
Stage 4 — Feature Scaling
Scaler is fitted on X_train only, then applied to both splits.

Strategies available:
  StandardScaler  — zero mean, unit variance (sensitive to outliers)
  RobustScaler    — median/IQR based (recommended when outliers remain)
  MinMaxScaler    — [0, 1] range (use when distribution shape matters)
  None            — passthrough (binary / already scaled columns)
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from utils.state import init
from utils.ui import stage_header


SCALER_MAP = {
    "RobustScaler (recommended — handles outliers)": RobustScaler,
    "StandardScaler (zero mean, unit std)":          StandardScaler,
    "MinMaxScaler ([0, 1] range)":                   MinMaxScaler,
    "None (passthrough — no scaling)":               None,
}


def show():
    init()
    stage_header("4_scaling", "Scale numeric features. Scaler fitted on train set only to prevent leakage.")

    if not st.session_state.get("df_encoded"):
        st.warning("Complete **Stage 3 · Encoding** first.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    all_cols = list(X_train.columns)

    # ── Step 1: Identify scalable columns ─────────────────────────────────────
    st.markdown("### Step 1 — Column overview")

    # Auto-detect: binary columns (only 0/1 or single unique value) → skip
    def _is_binary(col):
        uv = set(X_train[col].dropna().unique())
        return uv <= {0, 1}

    numeric_scalable = [c for c in all_cols
                        if pd.api.types.is_numeric_dtype(X_train[c]) and not _is_binary(c)]
    binary_cols      = [c for c in all_cols if _is_binary(c)]
    non_numeric      = [c for c in all_cols if not pd.api.types.is_numeric_dtype(X_train[c])]

    m1, m2, m3 = st.columns(3)
    m1.metric("Scalable numeric",  len(numeric_scalable))
    m2.metric("Binary (skip)",     len(binary_cols))
    m3.metric("Non-numeric (warn)", len(non_numeric))

    if non_numeric:
        st.warning(
            f"Non-numeric columns detected — encode them in Stage 3 first: "
            f"**{', '.join(non_numeric[:8])}**{'…' if len(non_numeric)>8 else ''}"
        )

    with st.expander("Column details", expanded=False):
        detail_df = pd.DataFrame([
            {"Column": c, "Category": "scalable",    "Min": X_train[c].min(), "Max": X_train[c].max(), "Mean": round(X_train[c].mean(), 2), "Std": round(X_train[c].std(), 2)}
            for c in numeric_scalable
        ] + [
            {"Column": c, "Category": "binary (skip)", "Min": "", "Max": "", "Mean": "", "Std": ""}
            for c in binary_cols
        ] + [
            {"Column": c, "Category": "non-numeric", "Min": "", "Max": "", "Mean": "", "Std": ""}
            for c in non_numeric
        ])
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=280)

    st.divider()

    # ── Step 2: Choose scaler ──────────────────────────────────────────────────
    st.markdown("### Step 2 — Choose scaler")

    scaler_choice = st.selectbox(
        "Scaler for all numeric columns",
        list(SCALER_MAP.keys()),
        index=0,
        help=(
            "RobustScaler is the safest default after outlier treatment. "
            "Use StandardScaler if you Winsorized all outliers. "
            "MinMaxScaler preserves relative distances but is sensitive to extreme values."
        ),
    )

    # Per-column override
    st.markdown("**Per-column overrides** (optional)")
    if "scale_overrides" not in st.session_state:
        st.session_state["scale_overrides"] = {}
    overrides = st.session_state["scale_overrides"]

    with st.expander("Override scaler per column", expanded=False):
        per_col_options = ["(use global)"] + list(SCALER_MAP.keys())
        for col in numeric_scalable:
            curr = overrides.get(col, "(use global)")
            if curr not in per_col_options:
                curr = "(use global)"
            overrides[col] = st.selectbox(
                col,
                per_col_options,
                index=per_col_options.index(curr),
                key=f"scale_ov_{col}",
            )
    st.session_state["scale_overrides"] = overrides

    st.divider()

    # ── Step 3: Columns to exclude ────────────────────────────────────────────
    st.markdown("### Step 3 — Exclude columns from scaling (optional)")
    exclude_cols = st.multiselect(
        "Hold these columns unscaled (e.g. already-scaled or binary-coded ordinals):",
        options=numeric_scalable,
        default=[],
        key="scale_exclude",
    )

    st.divider()

    # ── Step 4: Distribution preview ─────────────────────────────────────────
    st.markdown("### Step 4 — Distribution preview (train set, before scaling)")
    if numeric_scalable:
        preview_col = st.selectbox("Preview column", numeric_scalable, key="scale_prev_col")
        import plotly.express as px
        fig = px.histogram(X_train[preview_col].dropna(), nbins=30,
                           title=f"{preview_col} — distribution",
                           labels={"value": preview_col})
        fig.update_layout(height=250, margin=dict(t=35, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Step 5: Apply ─────────────────────────────────────────────────────────
    st.markdown("### Step 5 — Apply scaling")

    if st.button("Apply scaling", type="primary", key="apply_scale"):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        scale_log = []
        scalers_used = {}   # col → fitted scaler object (for inverse-transform later)

        global_cls = SCALER_MAP[scaler_choice]

        for col in numeric_scalable:
            if col in exclude_cols:
                scale_log.append(f"  {col}: excluded (passthrough)")
                continue

            override = overrides.get(col, "(use global)")
            if override != "(use global)":
                cls = SCALER_MAP.get(override)
            else:
                cls = global_cls

            if cls is None:
                scale_log.append(f"  {col}: passthrough (None)")
                continue

            scaler = cls()
            X_tr[[col]] = scaler.fit_transform(X_tr[[col]])
            X_te[[col]] = scaler.transform(X_te[[col]])
            scalers_used[col] = scaler

        st.session_state["X_train"]      = X_tr
        st.session_state["X_test"]       = X_te
        st.session_state["scalers_used"] = scalers_used
        st.session_state["df_scaled"]    = True

        st.success(f"Scaling applied to {len(scalers_used)} columns using {scaler_choice}.")
        with st.expander("Scaling log", expanded=False):
            for line in scale_log:
                st.caption(line)

        st.divider()

        # ── After summary ─────────────────────────────────────────────────────
        st.markdown("### Result")
        m1, m2 = st.columns(2)
        m1.metric("Train shape", f"{X_tr.shape[0]} × {X_tr.shape[1]}")
        m2.metric("Test shape",  f"{X_te.shape[0]} × {X_te.shape[1]}")

        st.markdown("**Scaled column statistics (train):**")
        scaled_cols = list(scalers_used.keys())
        if scaled_cols:
            stats = X_tr[scaled_cols].describe().T[["mean", "std", "min", "max"]].round(3)
            st.dataframe(stats, use_container_width=True)

        with st.expander("Preview scaled training data (first 5 rows)", expanded=True):
            st.dataframe(X_tr.head(), use_container_width=True)

        st.success("Ready for Stage 5 · Correlation Filter.")

    elif st.session_state.get("df_scaled"):
        X_tr = st.session_state["X_train"]
        X_te = st.session_state["X_test"]
        st.info(f"Scaling already applied — Train: {X_tr.shape[0]} × {X_tr.shape[1]}, "
                f"Test: {X_te.shape[0]} × {X_te.shape[1]}")
        if st.button("Re-apply scaling (resets current result)", key="reapply_scale"):
            st.session_state["df_scaled"] = False
            st.rerun()
