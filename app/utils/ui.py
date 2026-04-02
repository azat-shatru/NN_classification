"""Shared UI helpers used across all stage pages."""
import streamlit as st
import pandas as pd
import numpy as np

STAGE_LABELS = {
    "0_audit":       "Stage 0 · Data Audit",
    "1_missing":     "Stage 1 · Missing Values",
    "2_outliers":    "Stage 2 · Outlier Treatment",
    "2b_viz":        "Stage 2.5 · Visualisation Dashboard",
    "3_encoding":    "Stage 3 · Encoding",
    "4_scaling":     "Stage 4 · Scaling",
    "5_correlation": "Stage 5 · Correlation Filter",
    "6_rf":          "Stage 6 · RF Importance",
    "7_factor":      "Stage 7 · Factor Analysis",
    "8_combtest":    "Stage 8 · Combination Testing",
    "9_nn":          "Stage 9 · Neural Network",
}

def stage_header(stage_key: str, subtitle: str = ""):
    label = STAGE_LABELS.get(stage_key, stage_key)
    st.markdown(f"## {label}")
    if subtitle:
        st.caption(subtitle)
    st.divider()

def require_data(key: str = "df") -> bool:
    """Return True if data is loaded; show a warning and return False otherwise."""
    if st.session_state.get(key) is None:
        st.warning("Upload a dataset on **Stage 0 · Data Audit** first.")
        return False
    return True


def drop_columns_panel(stage_key: str, label: str = "Drop columns before this stage"):
    """
    Reusable column-drop panel.
    Works on X_train/X_test if the split has been done (Stage 1 onward),
    otherwise works on df directly.

    Call this at the TOP of any stage's show() function.
    Returns True if columns were dropped (caller may want to st.rerun()).
    """
    split_done = st.session_state.get("X_train") is not None

    if split_done:
        ref_df = st.session_state["X_train"]
    elif st.session_state.get("df") is not None:
        target = st.session_state.get("target_col", "")
        ref_df = st.session_state["df"].drop(
            columns=[target] if target and target in st.session_state["df"].columns else [],
            errors="ignore"
        )
    else:
        return False

    all_cols = list(ref_df.columns)
    if not all_cols:
        return False

    state_key = f"drop_panel_{stage_key}"

    with st.expander(f"🗑 {label} ({len(all_cols)} columns currently)", expanded=False):
        # ── Quick-stats table ─────────────────────────────────────────────
        stats_rows = []
        for col in all_cols:
            s = ref_df[col]
            miss_pct = round(s.isnull().sum() / max(len(s), 1) * 100, 1)
            is_num = pd.api.types.is_numeric_dtype(s)
            non_null = pd.to_numeric(s.dropna(), errors="coerce").dropna()
            variance = round(float(non_null.var()), 4) if is_num and len(non_null) > 1 else None
            zero_pct = round(float((non_null == 0).sum() / max(len(non_null), 1) * 100), 1) \
                       if is_num and len(non_null) > 0 else None
            n_unique = s.nunique()
            stats_rows.append({
                "Column":    col,
                "Missing %": miss_pct,
                "Variance":  variance,
                "Zero %":    zero_pct,
                "Unique":    n_unique,
            })

        stats_df = pd.DataFrame(stats_rows)

        # ── Quick-filter buttons ──────────────────────────────────────────
        qb1, qb2, qb3 = st.columns(3)
        var_thr  = qb1.number_input("Variance threshold", 0.0, 10.0, 0.01,
                                    format="%.4f", key=f"{state_key}_vt")
        miss_thr = qb2.number_input("Missing % threshold", 0, 100, 50,
                                    key=f"{state_key}_mt")
        zero_thr = qb3.number_input("Zero % threshold", 0, 100, 90,
                                    key=f"{state_key}_zt")

        auto_low_var  = stats_df.loc[stats_df["Variance"].fillna(0)  < var_thr,  "Column"].tolist()
        auto_hi_miss  = stats_df.loc[stats_df["Missing %"] >= miss_thr, "Column"].tolist()
        auto_hi_zero  = stats_df.loc[stats_df["Zero %"].fillna(0)   >= zero_thr, "Column"].tolist()

        ab1, ab2, ab3, ab4 = st.columns(4)
        curr_drop = list(st.session_state.get(state_key, []))

        if ab1.button(f"Add low-variance ({len(auto_low_var)})",
                      key=f"{state_key}_btn_var", disabled=not auto_low_var):
            st.session_state[state_key] = list(set(curr_drop) | set(auto_low_var))
            st.rerun()

        if ab2.button(f"Add high-missing ({len(auto_hi_miss)})",
                      key=f"{state_key}_btn_miss", disabled=not auto_hi_miss):
            st.session_state[state_key] = list(set(curr_drop) | set(auto_hi_miss))
            st.rerun()

        if ab3.button(f"Add high-zero ({len(auto_hi_zero)})",
                      key=f"{state_key}_btn_zero", disabled=not auto_hi_zero):
            st.session_state[state_key] = list(set(curr_drop) | set(auto_hi_zero))
            st.rerun()

        if ab4.button("Clear selection", key=f"{state_key}_clear"):
            st.session_state[state_key] = []
            st.rerun()

        # ── Column stats preview ──────────────────────────────────────────
        with st.expander("View column stats", expanded=False):
            # Highlight flagged rows
            def _highlight(row):
                flags = []
                if row["Variance"] is not None and row["Variance"] < var_thr:
                    flags.append("background-color:#fff3cd")
                if row["Missing %"] >= miss_thr:
                    flags.append("background-color:#f8d7da")
                if row["Zero %"] is not None and row["Zero %"] >= zero_thr:
                    flags.append("background-color:#fff3cd")
                color = flags[0] if flags else ""
                return [color] * len(row)

            st.dataframe(
                stats_df.style.apply(_highlight, axis=1),
                use_container_width=True, hide_index=True, height=220
            )

        # ── Manual multiselect — driven entirely by session state ─────────
        # Use on_change callback to write back immediately
        def _on_multiselect_change():
            st.session_state[state_key] = st.session_state[f"{state_key}_sel"]

        queued = [c for c in st.session_state.get(state_key, []) if c in all_cols]

        st.multiselect(
            "Columns to drop (edit freely):",
            options=all_cols,
            default=queued,
            key=f"{state_key}_sel",
            on_change=_on_multiselect_change,
        )

        # Read back current selection from state (always up-to-date)
        to_drop = [c for c in st.session_state.get(state_key, []) if c in all_cols]

        # ── Live status ───────────────────────────────────────────────────
        if to_drop:
            st.warning(
                f"**{len(to_drop)} queued for removal** — "
                f"**{len(all_cols) - len(to_drop)} columns** will remain after apply."
            )
        else:
            st.caption("No columns selected for removal.")

        # ── Apply ─────────────────────────────────────────────────────────
        if st.button("Apply drops now", type="primary",
                     key=f"{state_key}_apply", disabled=not to_drop):
            drop_set = set(to_drop)

            if split_done:
                X_tr = st.session_state["X_train"].drop(columns=drop_set, errors="ignore")
                X_te = st.session_state["X_test"].drop(columns=drop_set, errors="ignore")
                st.session_state["X_train"] = X_tr
                st.session_state["X_test"]  = X_te
            else:
                df_new = st.session_state["df"].drop(columns=drop_set, errors="ignore")
                st.session_state["df"] = df_new

            for key in ["numeric_cols", "categorical_cols", "ordinal_cols"]:
                st.session_state[key] = [
                    c for c in (st.session_state.get(key) or []) if c not in drop_set
                ]

            st.session_state[state_key] = []
            st.rerun()   # immediate full refresh

    return False
