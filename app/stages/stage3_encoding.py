"""
Stage 3 — Categorical Encoding
Fits encoders on train only, then applies to both splits.

Encoding strategies:
  Binary      → keep as-is (already 0/1)
  One-Hot     → pd.get_dummies (low-cardinality categoricals)
  Target Enc  → mean(target) per category (high-cardinality)
  Ordinal Enc → integer rank per user-defined category order
  Label Enc   → arbitrary integer codes (no order implied)
  Passthrough → numeric columns, skip
"""
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

from utils.state import init
from utils.ui import stage_header


# ── encoding helpers ──────────────────────────────────────────────────────────

def _one_hot(X_train: pd.DataFrame, X_test: pd.DataFrame, cols: list) -> tuple:
    """One-hot encode selected columns. Returns (X_train_new, X_test_new, new_col_names)."""
    X_tr = pd.get_dummies(X_train, columns=cols, drop_first=False, dtype=int)
    # Align test to train columns (fill missing dummies with 0)
    X_te = pd.get_dummies(X_test, columns=cols, drop_first=False, dtype=int)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    new_cols = [c for c in X_tr.columns if c not in X_train.columns or c in cols]
    return X_tr, X_te


def _target_encode(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, cols: list) -> tuple:
    """Replace each category with its mean target value (fitted on train only)."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    maps = {}
    global_mean = float(pd.to_numeric(y_train, errors="coerce").mean())

    for col in cols:
        tmp = X_train[[col]].copy()
        tmp["__y"] = y_train.values
        enc_map = tmp.groupby(col)["__y"].mean().to_dict()
        maps[col] = enc_map
        X_tr[col] = X_train[col].map(enc_map).fillna(global_mean)
        X_te[col] = X_test[col].map(enc_map).fillna(global_mean)

    return X_tr, X_te, maps


def _ordinal_encode(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    col: str, order: list) -> tuple:
    """Encode col according to user-supplied order list → integers 0,1,2,..."""
    rank_map = {v: i for i, v in enumerate(order)}
    fallback = len(order)   # unseen categories → one beyond last rank
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[col] = X_train[col].map(rank_map).fillna(fallback).astype(int)
    X_te[col] = X_test[col].map(rank_map).fillna(fallback).astype(int)
    return X_tr, X_te


def _label_encode(X_train: pd.DataFrame, X_test: pd.DataFrame, col: str) -> tuple:
    """Assign arbitrary integer codes. Unseen test categories → -1."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    cats = sorted(X_train[col].dropna().unique().tolist(), key=str)
    cat_map = {v: i for i, v in enumerate(cats)}
    X_tr[col] = X_train[col].map(cat_map)
    X_te[col] = X_test[col].map(cat_map).fillna(-1).astype(int)
    return X_tr, X_te


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("3_encoding", "Encode categorical/ordinal columns. Encoders are fitted on train set only.")

    # Require Stage 1 to have run
    if st.session_state.get("X_train") is None:
        st.warning("Complete **Stage 1 · Missing Values** (including the train/test split) first.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]
    target  = st.session_state.get("target_col", "target")

    num_cols = list(st.session_state.get("numeric_cols") or [])
    cat_cols = list(st.session_state.get("categorical_cols") or [])
    ord_cols = list(st.session_state.get("ordinal_cols") or [])

    # Limit to columns actually in X_train (dropped cols removed in Stage 1)
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    ord_cols = [c for c in ord_cols if c in X_train.columns]

    # Columns with no type assigned — auto-classify
    unassigned = [c for c in X_train.columns
                  if c not in num_cols + cat_cols + ord_cols]
    if unassigned:
        auto_num = [c for c in unassigned if pd.api.types.is_numeric_dtype(X_train[c])]
        auto_cat = [c for c in unassigned if c not in auto_num]
        num_cols += auto_num
        cat_cols += auto_cat

    st.markdown("### Step 1 — Column type summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric", len(num_cols))
    c2.metric("Categorical", len(cat_cols))
    c3.metric("Ordinal", len(ord_cols))

    with st.expander("View column assignments", expanded=False):
        type_df = pd.DataFrame([
            {"Column": c, "Type": "numeric"} for c in num_cols
        ] + [
            {"Column": c, "Type": "categorical"} for c in cat_cols
        ] + [
            {"Column": c, "Type": "ordinal"} for c in ord_cols
        ])
        st.dataframe(type_df, use_container_width=True, height=250)
        st.caption("Change column types back in Stage 0 · Data Audit if needed.")

    st.divider()

    # ── Step 2: Encoding strategy per column ──────────────────────────────────
    st.markdown("### Step 2 — Choose encoding strategy")
    st.caption(
        "Binary columns (≤2 unique values) are kept as-is. "
        "Numeric columns are passed through unchanged."
    )

    OHE_THRESHOLD = st.slider(
        "One-Hot cardinality threshold (auto-suggest OHE if unique values ≤ this)",
        2, 30, 10, key="ohe_thresh"
    )

    # Per-column strategy state
    if "enc_strategies" not in st.session_state:
        st.session_state["enc_strategies"] = {}
    if "ord_orders" not in st.session_state:
        st.session_state["ord_orders"] = {}

    strats = st.session_state["enc_strategies"]
    ord_orders = st.session_state["ord_orders"]

    # ── Categorical columns ───────────────────────────────────────────────────
    if cat_cols:
        st.markdown("#### Categorical columns")
        cat_rows = []
        for col in cat_cols:
            n_unique = X_train[col].nunique()
            is_binary = n_unique <= 2
            default = (
                "Binary (keep)"    if is_binary else
                "One-Hot"          if n_unique <= OHE_THRESHOLD else
                "Target Encoding"
            )
            strats[col] = default if col not in strats else strats[col]
            cat_rows.append({
                "Column": col,
                "Unique values": n_unique,
                "Suggested": default,
            })

        cat_tbl = pd.DataFrame(cat_rows)
        st.dataframe(cat_tbl, use_container_width=True, hide_index=True)

        st.markdown("**Override strategies:**")
        for col in cat_cols:
            n_unique = X_train[col].nunique()
            is_binary = n_unique <= 2
            options = (
                ["Binary (keep)"]
                if is_binary else
                ["One-Hot", "Target Encoding", "Label Encoding", "Binary (keep)"]
            )
            curr = strats.get(col, options[0])
            if curr not in options:
                curr = options[0]
            strats[col] = st.selectbox(
                f"{col}  ({n_unique} unique)",
                options,
                index=options.index(curr),
                key=f"enc_cat_{col}",
            )

    # ── Ordinal columns ───────────────────────────────────────────────────────
    if ord_cols:
        st.markdown("#### Ordinal columns")
        for col in ord_cols:
            n_unique = X_train[col].nunique()
            is_binary = n_unique <= 2
            options = (
                ["Binary (keep)"]
                if is_binary else
                ["Ordinal Encoding", "Label Encoding", "One-Hot", "Target Encoding"]
            )
            curr = strats.get(col, options[0])
            if curr not in options:
                curr = options[0]
            strats[col] = st.selectbox(
                f"{col}  ({n_unique} unique)",
                options,
                index=options.index(curr),
                key=f"enc_ord_{col}",
            )
            if strats[col] == "Ordinal Encoding":
                cats = sorted(X_train[col].dropna().unique().tolist(), key=str)
                curr_order = ord_orders.get(col, cats)
                # Keep only values still in train
                curr_order = [v for v in curr_order if v in cats] + \
                             [v for v in cats if v not in curr_order]
                st.caption(
                    f"Define category order for **{col}** "
                    f"(drag-and-drop not available — edit the text field below)."
                )
                order_str = st.text_area(
                    f"Category order — one per line (lowest → highest)",
                    value="\n".join(str(v) for v in curr_order),
                    height=min(150, len(cats) * 22 + 30),
                    key=f"ord_order_{col}",
                )
                ord_orders[col] = [v.strip() for v in order_str.splitlines() if v.strip()]

    st.session_state["enc_strategies"] = strats
    st.session_state["ord_orders"]     = ord_orders

    st.divider()

    # ── Step 3: Preview ───────────────────────────────────────────────────────
    st.markdown("### Step 3 — Preview encoding plan")
    plan_rows = []
    for col in cat_cols + ord_cols:
        plan_rows.append({
            "Column":   col,
            "Type":     "categorical" if col in cat_cols else "ordinal",
            "Strategy": strats.get(col, "—"),
            "Notes":    (f"order defined ({len(ord_orders.get(col, []))} values)"
                         if strats.get(col) == "Ordinal Encoding"
                         else ""),
        })
    if plan_rows:
        st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
    if num_cols:
        st.info(f"**{len(num_cols)} numeric columns** will be passed through unchanged: "
                f"{', '.join(num_cols[:8])}{'…' if len(num_cols)>8 else ''}")

    st.divider()

    # ── Step 4: Apply ─────────────────────────────────────────────────────────
    st.markdown("### Step 4 — Apply encoding")

    if st.button("Apply encoding", type="primary", key="apply_enc"):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        enc_log = []

        # Process all columns that need encoding
        all_enc_cols = cat_cols + ord_cols

        # Group by strategy for batch operations
        ohe_cols    = [c for c in all_enc_cols if strats.get(c) == "One-Hot"]
        tgt_cols    = [c for c in all_enc_cols if strats.get(c) == "Target Encoding"]
        lbl_cols    = [c for c in all_enc_cols if strats.get(c) == "Label Encoding"]
        ord_enc_cols = [c for c in all_enc_cols if strats.get(c) == "Ordinal Encoding"]
        # Binary / passthrough — no action needed

        if ohe_cols:
            before_ncols = X_tr.shape[1]
            X_tr, X_te = _one_hot(X_tr, X_te, ohe_cols)
            added = X_tr.shape[1] - before_ncols
            enc_log.append(f"One-Hot: {ohe_cols} → +{added} dummy columns")

        if tgt_cols:
            X_tr, X_te, tgt_maps = _target_encode(X_tr, X_te, y_train, tgt_cols)
            st.session_state["target_enc_maps"] = tgt_maps
            enc_log.append(f"Target Encoding: {tgt_cols}")

        for col in lbl_cols:
            X_tr, X_te = _label_encode(X_tr, X_te, col)
            enc_log.append(f"Label Encoding: {col}")

        for col in ord_enc_cols:
            order = ord_orders.get(col, [])
            if not order:
                order = sorted(X_train[col].dropna().unique().tolist(), key=str)
            X_tr, X_te = _ordinal_encode(X_tr, X_te, col, order)
            enc_log.append(f"Ordinal Encoding: {col} ({len(order)} levels)")

        # Persist
        st.session_state["X_train"]       = X_tr
        st.session_state["X_test"]        = X_te
        st.session_state["df_encoded"]    = True

        # Update column type lists to reflect new encoded column set
        # After OHE, original cat cols are replaced with dummy cols
        ohe_dummy_cols = [c for c in X_tr.columns
                          if any(c.startswith(f"{oc}_") for oc in ohe_cols)]
        remaining_cat  = [c for c in cat_cols if c not in ohe_cols and c in X_tr.columns]
        remaining_ord  = [c for c in ord_cols if c in X_tr.columns]
        st.session_state["numeric_cols"]     = list(X_tr.columns)  # all cols now numeric-ish
        st.session_state["categorical_cols"] = []
        st.session_state["ordinal_cols"]     = []

        st.success("Encoding applied successfully!")
        for log_line in enc_log:
            st.caption(f"• {log_line}")

        st.divider()

        # ── After summary ─────────────────────────────────────────────────────
        st.markdown("### Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train rows", X_tr.shape[0])
        m2.metric("Train columns (after encoding)", X_tr.shape[1])
        m3.metric("Test columns", X_te.shape[1])

        # Show first few rows of encoded train set
        with st.expander("Preview encoded training data (first 5 rows)", expanded=True):
            st.dataframe(X_tr.head(), use_container_width=True)

        # Check for any remaining non-numeric columns
        non_numeric = [c for c in X_tr.columns
                       if not pd.api.types.is_numeric_dtype(X_tr[c])]
        if non_numeric:
            st.warning(
                f"These columns are still non-numeric after encoding — "
                f"review their strategies: **{', '.join(non_numeric)}**"
            )
        else:
            st.success("All columns are numeric. Ready for Stage 4 · Scaling.")

    elif st.session_state.get("df_encoded"):
        X_tr = st.session_state["X_train"]
        X_te = st.session_state["X_test"]
        st.info(f"Encoding already applied — Train: {X_tr.shape[0]} × {X_tr.shape[1]}, "
                f"Test: {X_te.shape[0]} × {X_te.shape[1]}")
        if st.button("Re-apply encoding (resets current result)", key="reapply_enc"):
            st.session_state["df_encoded"] = False
            st.rerun()
