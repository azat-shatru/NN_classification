"""
Stage 7 — Factor Analysis
Reduce remaining features via Exploratory Factor Analysis (EFA).
User can:
  - Name factors and use them as composite inputs
  - Or keep the individual high-loading features instead
  - Or skip factor analysis entirely

Requires: factor_analyzer  (pip install factor_analyzer)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.state import init
from utils.ui import stage_header


def _check_fa():
    try:
        from factor_analyzer import FactorAnalyzer
        return True
    except ImportError:
        return False


def show():
    init()
    stage_header("7_factor", "Exploratory Factor Analysis — group correlated features into latent factors.")

    if not st.session_state.get("rf_done"):
        st.warning("Complete **Stage 6 · RF Importance** first.")
        return

    if not _check_fa():
        st.error(
            "`factor_analyzer` package not found. "
            "Run:  `pip install factor_analyzer`  then restart the app."
        )
        return

    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    y_train = st.session_state["y_train"]

    numeric_cols = [c for c in X_train.columns
                    if pd.api.types.is_numeric_dtype(X_train[c])]
    X_num = X_train[numeric_cols].dropna()

    st.markdown(f"**{len(numeric_cols)} features** entering factor analysis.")

    # Skip option
    if st.checkbox("Skip factor analysis — proceed with current features", key="fa_skip"):
        st.session_state["factor_done"]       = True
        st.session_state["factor_features"]   = numeric_cols
        st.info("Skipped. Proceed to Stage 8 · Combination Testing.")
        return

    st.divider()

    # ── Step 1: Suitability tests ─────────────────────────────────────────────
    st.markdown("### Step 1 — Suitability tests")
    if st.button("Run Bartlett & KMO tests", key="fa_tests"):
        try:
            chi2, p = calculate_bartlett_sphericity(X_num)
            kmo_all, kmo_model = calculate_kmo(X_num)
            st.session_state["fa_bartlett"] = (chi2, p)
            st.session_state["fa_kmo"]      = (kmo_all, kmo_model)
        except Exception as e:
            st.error(f"Test error: {e}")

    if st.session_state.get("fa_bartlett"):
        chi2, p = st.session_state["fa_bartlett"]
        kmo_all, kmo_model = st.session_state["fa_kmo"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Bartlett χ²", f"{chi2:.1f}")
        m2.metric("Bartlett p-value", f"{p:.4f}",
                  delta="✓ significant" if p < 0.05 else "✗ not significant",
                  delta_color="normal")
        m3.metric("KMO score", f"{kmo_model:.3f}",
                  delta="✓ adequate (≥0.6)" if kmo_model >= 0.6 else "✗ inadequate",
                  delta_color="normal")

        if p >= 0.05:
            st.warning("Bartlett test not significant — variables may not be correlated enough for FA.")
        if kmo_model < 0.6:
            st.warning("KMO < 0.6 — factor analysis may not be suitable for this data.")

    st.divider()

    # ── Step 2: Choose number of factors ─────────────────────────────────────
    st.markdown("### Step 2 — Scree plot & number of factors")
    max_factors = min(len(numeric_cols) - 1, 15)

    if st.button("Compute scree plot (eigenvalues)", key="fa_scree"):
        fa_scree = FactorAnalyzer(n_factors=max_factors, rotation=None)
        fa_scree.fit(X_num)
        ev, _ = fa_scree.get_eigenvalues()
        st.session_state["fa_eigenvalues"] = ev[:max_factors].tolist()

    if st.session_state.get("fa_eigenvalues"):
        ev = st.session_state["fa_eigenvalues"]
        n_above_1 = sum(1 for e in ev if e >= 1.0)
        scree_fig = go.Figure()
        scree_fig.add_trace(go.Scatter(
            x=list(range(1, len(ev)+1)), y=ev,
            mode="lines+markers", name="Eigenvalue",
            marker=dict(size=8), line=dict(color="#2E75B6"),
        ))
        scree_fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                            annotation_text="Kaiser criterion (eigenvalue=1)")
        scree_fig.update_layout(
            title="Scree Plot", xaxis_title="Factor", yaxis_title="Eigenvalue",
            height=350, margin=dict(t=50, b=30),
        )
        st.plotly_chart(scree_fig, use_container_width=True)
        st.info(f"**{n_above_1} factors** have eigenvalue ≥ 1 (Kaiser criterion).")

    st.divider()

    # ── Step 3: Run FA ────────────────────────────────────────────────────────
    st.markdown("### Step 3 — Run factor analysis")

    default_n = st.session_state.get("fa_eigenvalues") and \
                sum(1 for e in st.session_state["fa_eigenvalues"] if e >= 1.0) or 3

    c1, c2 = st.columns(2)
    n_factors = c1.number_input("Number of factors", 1, max_factors, default_n, 1, key="fa_n")
    rotation  = c2.selectbox("Rotation", ["varimax", "promax", "oblimin", "none"],
                              key="fa_rotation")

    rotation_arg = None if rotation == "none" else rotation

    if st.button("Run Factor Analysis", type="primary", key="run_fa"):
        try:
            fa = FactorAnalyzer(n_factors=int(n_factors), rotation=rotation_arg)
            fa.fit(X_num)
            loadings = pd.DataFrame(
                fa.loadings_,
                index=numeric_cols,
                columns=[f"F{i+1}" for i in range(int(n_factors))],
            ).round(3)
            variance = pd.DataFrame(
                fa.get_factor_variance(),
                index=["SS Loadings", "Proportion Var", "Cumulative Var"],
                columns=[f"F{i+1}" for i in range(int(n_factors))],
            ).round(3)
            factor_scores_tr = fa.transform(X_num)
            factor_scores_te = fa.transform(X_test[numeric_cols].fillna(X_num.mean()))

            st.session_state["fa_model"]     = fa
            st.session_state["fa_loadings"]  = loadings
            st.session_state["fa_variance"]  = variance
            st.session_state["fa_scores_tr"] = factor_scores_tr
            st.session_state["fa_scores_te"] = factor_scores_te
            st.success(f"Factor analysis complete ({n_factors} factors, {rotation} rotation).")
        except Exception as e:
            st.error(f"FA error: {e}")

    if st.session_state.get("fa_loadings") is None:
        return

    loadings = st.session_state["fa_loadings"]
    variance = st.session_state["fa_variance"]

    st.divider()

    # ── Step 4: Loadings table ────────────────────────────────────────────────
    st.markdown("### Step 4 — Factor loadings")

    loading_thresh = st.slider("Highlight loadings above |λ| ≥", 0.3, 0.9, 0.4, 0.05,
                               key="fa_load_thresh")

    def _style_loadings(val):
        if pd.api.types.is_number(val) and abs(val) >= loading_thresh:
            return "background-color: #d6e4f7; font-weight: bold"
        return ""

    st.dataframe(
        loadings.style.applymap(_style_loadings),
        use_container_width=True,
        height=min(500, len(loadings) * 36 + 40),
    )

    st.markdown("**Variance explained:**")
    st.dataframe(variance, use_container_width=True)

    # ── Biplot (F1 vs F2) ──────────────────────────────────────────────────────
    if loadings.shape[1] >= 2:
        with st.expander("Biplot — F1 vs F2 loadings"):
            biplot_fig = px.scatter(
                loadings.reset_index(),
                x="F1", y="F2",
                text="index",
                title="Factor Biplot (F1 × F2)",
            )
            biplot_fig.update_traces(textposition="top center", marker_size=8)
            biplot_fig.add_hline(y=0, line_color="gray", line_dash="dot")
            biplot_fig.add_vline(x=0, line_color="gray", line_dash="dot")
            biplot_fig.update_layout(height=400, margin=dict(t=50, b=30))
            st.plotly_chart(biplot_fig, use_container_width=True)

    st.divider()

    # ── Step 5: Choose output mode ────────────────────────────────────────────
    st.markdown("### Step 5 — Choose output: factor scores or high-loading features")
    output_mode = st.radio(
        "What to pass to the next stage?",
        [
            "Use factor scores as new features (F1, F2, …)",
            "Keep high-loading original features only",
            "Keep both factors and original features",
        ],
        key="fa_output_mode",
    )

    factor_names = {}
    if "Use factor scores" in output_mode or "both" in output_mode:
        st.markdown("**Name your factors** (optional — for reporting):")
        cols_fn = st.columns(min(4, loadings.shape[1]))
        for i, fcol in enumerate(loadings.columns):
            factor_names[fcol] = cols_fn[i % 4].text_input(
                fcol, fcol, key=f"fa_name_{fcol}")

    high_load_features = []
    if "high-loading" in output_mode or "both" in output_mode:
        high_load_features = loadings.abs().max(axis=1)
        high_load_features = high_load_features[high_load_features >= loading_thresh].index.tolist()
        st.info(f"**{len(high_load_features)}** features have |loading| ≥ {loading_thresh} on at least one factor.")
        high_load_features = st.multiselect(
            "High-loading features to keep:",
            options=numeric_cols,
            default=high_load_features,
            key="fa_highload_sel",
        )

    st.divider()

    # ── Step 6: Apply ─────────────────────────────────────────────────────────
    st.markdown("### Step 6 — Apply")
    if st.button("Apply factor analysis results", type="primary", key="apply_fa"):
        fa_scores_tr = st.session_state["fa_scores_tr"]
        fa_scores_te = st.session_state["fa_scores_te"]
        factor_col_names = [factor_names.get(f, f) for f in loadings.columns]

        fa_tr_df = pd.DataFrame(fa_scores_tr, index=X_train.index, columns=factor_col_names)
        fa_te_df = pd.DataFrame(fa_scores_te, index=X_test.index,  columns=factor_col_names)

        if output_mode == "Use factor scores as new features (F1, F2, …)":
            X_tr_out = fa_tr_df
            X_te_out = fa_te_df
            out_features = factor_col_names

        elif output_mode == "Keep high-loading original features only":
            X_tr_out = X_train[high_load_features]
            X_te_out = X_test[high_load_features]
            out_features = high_load_features

        else:  # both
            X_tr_out = pd.concat([fa_tr_df, X_train[high_load_features]], axis=1)
            X_te_out = pd.concat([fa_te_df, X_test[high_load_features]], axis=1)
            out_features = factor_col_names + high_load_features

        # Deduplicate columns
        X_tr_out = X_tr_out.loc[:, ~X_tr_out.columns.duplicated()]
        X_te_out = X_te_out.loc[:, ~X_te_out.columns.duplicated()]
        out_features = list(dict.fromkeys(out_features))

        st.session_state["X_train"]          = X_tr_out
        st.session_state["X_test"]           = X_te_out
        st.session_state["factor_done"]      = True
        st.session_state["factor_features"]  = out_features

        st.success(
            f"Factor analysis applied — "
            f"**{X_tr_out.shape[1]} features** entering Stage 8."
        )
        with st.expander("Output features"):
            st.write(", ".join(out_features))

    elif st.session_state.get("factor_done"):
        ff = st.session_state.get("factor_features", [])
        st.info(f"Factor analysis done — **{len(ff)} features** ready for Stage 8.")
        if st.button("Redo factor analysis", key="redo_fa"):
            st.session_state["factor_done"] = False
            st.rerun()
