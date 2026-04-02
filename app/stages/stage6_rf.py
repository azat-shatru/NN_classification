"""
Stage 6 — Random Forest Feature Importance
Train a quick RF on the full feature set; rank features by importance.
User selects which features to keep for the next stages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

from utils.state import init
from utils.ui import stage_header


def show():
    init()
    stage_header("6_rf", "Rank features by Random Forest importance and select the most informative ones.")

    if not st.session_state.get("corr_done"):
        st.warning("Complete **Stage 5 · Correlation Filter** first.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    y_train = st.session_state["y_train"]

    numeric_cols = [c for c in X_train.columns
                    if pd.api.types.is_numeric_dtype(X_train[c])]
    X_tr = X_train[numeric_cols]

    st.markdown(f"**{len(numeric_cols)} features** after correlation filter.")

    st.divider()

    # ── Step 1: RF parameters ─────────────────────────────────────────────────
    st.markdown("### Step 1 — Random Forest settings")
    c1, c2, c3 = st.columns(3)
    n_trees    = c1.number_input("Number of trees", 50, 500, 200, 50)
    max_depth  = c2.number_input("Max depth (0 = unlimited)", 0, 20, 0, 1)
    rand_seed  = c3.number_input("Random seed", 0, 9999, 42, 1)

    max_depth_val = None if max_depth == 0 else int(max_depth)

    if st.button("Train Random Forest", type="primary", key="train_rf"):
        with st.spinner(f"Training RF with {n_trees} trees..."):
            rf = RandomForestClassifier(
                n_estimators=int(n_trees),
                max_depth=max_depth_val,
                random_state=int(rand_seed),
                n_jobs=-1,
            )
            rf.fit(X_tr, y_train)

        importances = pd.DataFrame({
            "Feature":    X_tr.columns,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        importances["Rank"]         = importances.index + 1
        importances["Cumulative %"] = (importances["Importance"].cumsum() /
                                       importances["Importance"].sum() * 100).round(1)

        st.session_state["rf_importances"] = importances
        st.session_state["rf_model"]       = rf
        st.session_state["rf_done"]        = False  # not finalised yet
        st.success("RF trained successfully.")

    if st.session_state.get("rf_importances") is None:
        return

    importances = st.session_state["rf_importances"]

    st.divider()

    # ── Step 2: Importance chart ──────────────────────────────────────────────
    st.markdown("### Step 2 — Feature importance chart")

    top_n = st.slider("Show top N features", 5, len(importances), min(30, len(importances)),
                      key="rf_topn")
    top_df = importances.head(top_n)

    fig = px.bar(
        top_df,
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        text=top_df["Importance"].map(lambda v: f"{v:.4f}"),
        title=f"Top {top_n} features by RF importance",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 22 + 80),
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cumulative importance curve ───────────────────────────────────────────
    with st.expander("Cumulative importance curve"):
        cum_fig = px.line(
            importances.head(top_n),
            x="Rank", y="Cumulative %",
            markers=True,
            title="Cumulative importance by feature rank",
            labels={"Cumulative %": "Cumulative importance (%)"},
        )
        cum_fig.add_hline(y=80, line_dash="dash", line_color="orange",
                          annotation_text="80%")
        cum_fig.add_hline(y=95, line_dash="dash", line_color="red",
                          annotation_text="95%")
        cum_fig.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(cum_fig, use_container_width=True)

    st.divider()

    # ── Step 3: Full importance table ─────────────────────────────────────────
    st.markdown("### Step 3 — Full importance table")
    st.dataframe(importances, use_container_width=True, hide_index=True, height=300)

    st.divider()

    # ── Step 4: Select features to keep ──────────────────────────────────────
    st.markdown("### Step 4 — Select features to keep")

    sel_method = st.radio(
        "Selection method",
        ["Top N by importance", "Cumulative importance threshold", "Manual selection"],
        horizontal=True,
        key="rf_sel_method",
    )

    if sel_method == "Top N by importance":
        keep_n = st.slider("Keep top N features", 2, len(importances),
                           min(20, len(importances)), key="rf_keep_n")
        suggested = importances.head(keep_n)["Feature"].tolist()

    elif sel_method == "Cumulative importance threshold":
        cum_thresh = st.slider("Keep features until cumulative importance reaches:",
                               50, 100, 90, 5, key="rf_cum_thresh")
        mask = importances["Cumulative %"] <= cum_thresh
        # Always keep at least 1 feature
        if not mask.any():
            mask.iloc[0] = True
        suggested = importances.loc[mask, "Feature"].tolist()
        st.info(f"Keeps **{len(suggested)}** features reaching {cum_thresh}% cumulative importance.")

    else:  # Manual
        suggested = importances["Feature"].tolist()

    keep_features = st.multiselect(
        "Features to keep (edit freely):",
        options=importances["Feature"].tolist(),
        default=suggested,
        key="rf_keep_sel",
    )

    st.info(f"Keeping **{len(keep_features)}** of {len(importances)} features.")

    st.divider()

    # ── Step 5: Apply ─────────────────────────────────────────────────────────
    st.markdown("### Step 5 — Apply feature selection")
    if st.button("Apply RF feature selection", type="primary", key="apply_rf"):
        if not keep_features:
            st.error("Select at least one feature.")
        else:
            X_tr_new = X_train[keep_features]
            X_te_new = X_test[keep_features]
            st.session_state["X_train"]             = X_tr_new
            st.session_state["X_test"]              = X_te_new
            st.session_state["rf_selected_features"] = keep_features
            st.session_state["rf_done"]             = True

            st.success(
                f"RF selection applied — **{len(keep_features)} features** kept. "
                "Proceed to Stage 7 · Factor Analysis."
            )
            with st.expander("Selected features"):
                st.write(", ".join(keep_features))

    elif st.session_state.get("rf_done"):
        rf_sel = st.session_state.get("rf_selected_features", [])
        st.info(f"RF selection done — **{len(rf_sel)} features** retained.")
        if st.button("Redo RF selection", key="redo_rf"):
            st.session_state["rf_done"] = False
            st.rerun()
