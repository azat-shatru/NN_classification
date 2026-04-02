"""
Stage 1 — Missing Value Treatment
Runs AFTER train/test split to prevent data leakage.
Imputer is fitted on train rows only, then applied to both splits.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

from utils.state import init
from utils.ui import stage_header, require_data, drop_columns_panel


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_df(df, target, test_size=0.2, seed=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def _apply_imputer(X_train, X_test, cols, strategy, n_neighbors=5):
    if strategy == "knn":
        imp = KNNImputer(n_neighbors=n_neighbors)
    else:
        imp = SimpleImputer(strategy=strategy)

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[cols] = imp.fit_transform(X_train[cols])   # fit on train ONLY
    X_test[cols]  = imp.transform(X_test[cols])        # transform test
    return X_train, X_test


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("1_missing", "Train/test split happens here — imputation is fitted on train only.")

    if not require_data():
        return

    df     = st.session_state["df"].copy()
    target = st.session_state["target_col"]

    if target is None:
        st.warning("Set a target column in Stage 0 first.")
        return

    # ── Column drop panel ─────────────────────────────────────────────────────
    if drop_columns_panel("stage1", "Drop columns before missing-value treatment"):
        st.rerun()

    st.divider()

    # ── Step 1: Train / test split ────────────────────────────────────────────
    st.markdown("### Step 1 — Train / Test Split")
    st.info("Split happens **before** imputation to prevent data leakage.")

    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    seed      = st.number_input("Random seed", value=42, step=1)

    if st.button("Perform split"):
        X_tr, X_te, y_tr, y_te = _split_df(df, target, test_size, seed)
        st.session_state["X_train"] = X_tr
        st.session_state["X_test"]  = X_te
        st.session_state["y_train"] = y_tr
        st.session_state["y_test"]  = y_te
        st.success(f"Split complete — Train: {len(X_tr)} rows | Test: {len(X_te)} rows")

    if st.session_state.get("X_train") is None:
        st.warning("Perform the split above before proceeding.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()

    c1, c2 = st.columns(2)
    c1.metric("Train rows", len(X_train))
    c2.metric("Test rows",  len(X_test))

    st.divider()

    # ── Step 2: Missing values per column ─────────────────────────────────────
    st.markdown("### Step 2 — Missing value summary (train set)")

    miss = X_train.isnull().sum()
    miss_pct = (miss / len(X_train) * 100).round(1)
    miss_df = pd.DataFrame({
        "Column":        miss.index,
        "Missing count": miss.values,
        "Missing %":     miss_pct.values,
    }).query("`Missing count` > 0").sort_values("Missing %", ascending=False)

    if miss_df.empty:
        st.success("No missing values in the train set. Proceed to Stage 2.")
        st.session_state["df_imputed"] = True
        return

    st.dataframe(miss_df.reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── Step 3: Drop high-missing columns ─────────────────────────────────────
    st.markdown("### Step 3 — Drop columns with too many missing values")
    drop_thresh = st.slider("Drop columns with missing % above:", 10, 90, 40, 5)

    cols_to_drop = miss_df.loc[miss_df["Missing %"] > drop_thresh, "Column"].tolist()
    if cols_to_drop:
        st.warning(f"Columns flagged for dropping: **{', '.join(cols_to_drop)}**")
        confirmed_drop = st.multiselect(
            "Confirm columns to drop (deselect to keep):",
            options=cols_to_drop, default=cols_to_drop
        )
        if confirmed_drop:
            X_train = X_train.drop(columns=confirmed_drop)
            X_test  = X_test.drop(columns=confirmed_drop)
            # Update column type lists
            for key in ["numeric_cols", "categorical_cols", "ordinal_cols"]:
                updated = [c for c in (st.session_state.get(key) or [])
                           if c not in confirmed_drop]
                st.session_state[key] = updated
            st.success(f"Dropped: {confirmed_drop}")
    else:
        st.success(f"No columns exceed the {drop_thresh}% threshold.")

    st.divider()

    # ── Step 4: Imputation strategy per type ──────────────────────────────────
    st.markdown("### Step 4 — Imputation strategy")

    num_cols = [c for c in (st.session_state["numeric_cols"] or [])
                if c in X_train.columns and X_train[c].isnull().any()]
    cat_cols = [c for c in
                (st.session_state["categorical_cols"] or []) +
                (st.session_state["ordinal_cols"] or [])
                if c in X_train.columns and X_train[c].isnull().any()]

    num_strategy = cat_strategy = None

    if num_cols:
        st.markdown(f"**Numeric columns with missing values:** {', '.join(num_cols)}")
        num_strategy = st.selectbox(
            "Numeric imputation strategy",
            ["median", "mean", "knn"],
            index=0,
            help="Median is robust to outliers. KNN uses neighbouring rows."
        )
        if num_strategy == "knn":
            knn_k = st.slider("KNN neighbours (k)", 3, 15, 5)

    if cat_cols:
        st.markdown(f"**Categorical/Ordinal columns with missing values:** {', '.join(cat_cols)}")
        cat_strategy = st.selectbox(
            "Categorical imputation strategy",
            ["most_frequent", "constant"],
            index=0,
            help="most_frequent = mode. constant = fills with 'Unknown'."
        )

    st.divider()

    # ── Step 5: Apply imputation ──────────────────────────────────────────────
    st.markdown("### Step 5 — Apply imputation")

    if st.button("Apply imputation", type="primary"):
        if num_cols and num_strategy:
            k = knn_k if num_strategy == "knn" else 5
            X_train, X_test = _apply_imputer(X_train, X_test, num_cols,
                                             num_strategy, n_neighbors=k)

        if cat_cols and cat_strategy:
            fill_val = "Unknown" if cat_strategy == "constant" else None
            imp = __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(
                strategy=cat_strategy,
                fill_value=fill_val
            )
            X_train[cat_cols] = imp.fit_transform(X_train[cat_cols])
            X_test[cat_cols]  = imp.transform(X_test[cat_cols])

        # Persist imputed splits
        st.session_state["X_train"]     = X_train
        st.session_state["X_test"]      = X_test
        st.session_state["df_imputed"]  = True

        # Verify
        remaining = X_train.isnull().sum().sum()
        if remaining == 0:
            st.success("All missing values resolved. Proceed to Stage 2 · Outlier Treatment.")
        else:
            st.warning(f"{remaining} missing values remain — review the strategy above.")
