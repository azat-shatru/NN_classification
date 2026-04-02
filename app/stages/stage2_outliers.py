"""
Stage 2 — Outlier Detection & Treatment
Detects outliers per numeric column using IQR and Z-score,
multivariate outliers using Isolation Forest.
User decides treatment per column: Winsorize / Remove / Log-transform / Keep.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from scipy import stats

from utils.state import init
from utils.ui import stage_header, require_data, drop_columns_panel


# ── helpers ───────────────────────────────────────────────────────────────────

def _iqr_outliers(series: pd.Series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def _zscore_outliers(series: pd.Series, threshold=3.0):
    z = np.abs(stats.zscore(series.dropna()))
    idx = series.dropna().index[z > threshold]
    mask = pd.Series(False, index=series.index)
    mask[idx] = True
    return mask


def _winsorize(series: pd.Series, lower_p=1, upper_p=99):
    lo = np.percentile(series.dropna(), lower_p)
    hi = np.percentile(series.dropna(), upper_p)
    return series.clip(lower=lo, upper=hi)


def _log_transform(series: pd.Series):
    shift = max(0, -series.min()) + 1
    return np.log1p(series + shift)


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("2_outliers", "Detect and treat outliers in numeric columns.")

    if not require_data():
        return
    if not st.session_state.get("df_imputed"):
        st.warning("Complete **Stage 1 · Missing Values** first.")
        return

    # ── Column drop panel ─────────────────────────────────────────────────────
    if drop_columns_panel("stage2", "Drop columns before outlier treatment"):
        st.rerun()

    st.divider()

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    num_cols = [c for c in (st.session_state["numeric_cols"] or [])
                if c in X_train.columns]

    if not num_cols:
        st.info("No numeric columns found. Proceed to Stage 2.5.")
        return

    # ── 1. Column-level boxplots + IQR/Z-score ────────────────────────────────
    st.markdown("### Per-column outlier inspection")
    detection = st.radio("Detection method", ["IQR (1.5×)", "Z-score (threshold=3)"],
                         horizontal=True)

    treatments = {}   # col → treatment choice

    for col in num_cols:
        if detection.startswith("IQR"):
            mask, lo, hi = _iqr_outliers(X_train[col])
        else:
            mask = _zscore_outliers(X_train[col])

        n_out = mask.sum()
        pct   = n_out / len(X_train) * 100

        with st.expander(f"**{col}** — {n_out} outliers ({pct:.1f}%)", expanded=(n_out > 0)):
            c1, c2 = st.columns([2, 1])

            with c1:
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=X_train[col], name=col,
                    boxpoints="outliers", marker_color="#2E75B6",
                    line_color="#1F3864"
                ))
                fig.update_layout(height=280, margin=dict(t=20, b=20),
                                  yaxis_title=col)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.metric("Outlier count", n_out)
                st.metric("Outlier %", f"{pct:.1f}%")
                skew = X_train[col].skew()
                st.metric("Skewness", f"{skew:.2f}")

                choice = st.selectbox(
                    "Treatment", ["Keep", "Winsorize (1–99 pct)", "Remove rows", "Log-transform"],
                    key=f"treatment_{col}"
                )
                treatments[col] = choice

    st.divider()

    # ── 2. Multivariate — Isolation Forest ────────────────────────────────────
    st.markdown("### Multivariate outlier detection — Isolation Forest")
    st.caption("Detects rows that are anomalous across ALL numeric columns combined.")

    contamination = st.slider("Expected outlier fraction (%)", 1, 20, 5) / 100
    if st.button("Run Isolation Forest"):
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_train[num_cols].fillna(X_train[num_cols].median()))
        iso_mask = preds == -1
        st.session_state["iso_outlier_mask"] = pd.Series(iso_mask, index=X_train.index)

        n_iso = iso_mask.sum()
        st.warning(f"Isolation Forest flagged **{n_iso} rows** ({n_iso/len(X_train)*100:.1f}%) as outliers.")

        iso_remove = st.checkbox(f"Remove these {n_iso} rows from training set", value=False)
        if iso_remove:
            X_train = X_train[~iso_mask].reset_index(drop=True)
            y_train = st.session_state["y_train"][~iso_mask.values].reset_index(drop=True)
            st.session_state["y_train"] = y_train
            st.success(f"Removed {n_iso} multivariate outlier rows from train set.")

    st.divider()

    # ── 3. Apply column-level treatments ─────────────────────────────────────
    st.markdown("### Apply treatments")

    if st.button("Apply all treatments", type="primary"):
        removed_masks = []

        for col, choice in treatments.items():
            if choice == "Keep":
                continue
            elif choice == "Winsorize (1–99 pct)":
                X_train[col] = _winsorize(X_train[col])
                X_test[col]  = _winsorize(X_test[col])
            elif choice == "Log-transform":
                X_train[col] = _log_transform(X_train[col])
                X_test[col]  = _log_transform(X_test[col])
            elif choice == "Remove rows":
                if detection.startswith("IQR"):
                    mask, _, _ = _iqr_outliers(X_train[col])
                else:
                    mask = _zscore_outliers(X_train[col])
                removed_masks.append(mask)

        # Apply row removals (train only)
        if removed_masks:
            combined = pd.concat(removed_masks, axis=1).any(axis=1)
            n_removed = combined.sum()
            X_train = X_train[~combined].reset_index(drop=True)
            y_train = st.session_state["y_train"].reset_index(drop=True)
            y_train = y_train[~combined.values].reset_index(drop=True)
            st.session_state["y_train"] = y_train
            st.info(f"Removed {n_removed} rows flagged as outliers.")

        st.session_state["X_train"]       = X_train
        st.session_state["X_test"]        = X_test
        st.session_state["df_outliers_done"] = True
        st.success("Outlier treatment applied. Proceed to Stage 2.5 · Visualisation Dashboard.")

        # Preview after treatment
        st.markdown("#### Distribution after treatment")
        cols_changed = [c for c, t in treatments.items() if t != "Keep"]
        if cols_changed:
            fig = px.box(X_train[cols_changed], title="Post-treatment distributions")
            st.plotly_chart(fig, use_container_width=True)
