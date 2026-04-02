"""
Stage 8 — Combination Testing
Try different feature subsets, score each on accuracy / AUC / McFadden R²,
and select the best combination for the final NN.

Three modes:
  1. Forward stepwise   — start empty, add best feature each round
  2. Backward stepwise  — start full, remove worst feature each round
  3. Manual explorer    — user picks feature subsets and scores them
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import itertools, time

from utils.state import init
from utils.ui import stage_header


# ── scoring helper ─────────────────────────────────────────────────────────────

def _score_features(X_tr, y_tr, features, cv=5):
    """Quick cross-validated score using LogisticRegression as proxy model."""
    if not features:
        return {}
    Xf = X_tr[features]
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=500, random_state=42, multi_class="multinomial",
                            solver="lbfgs")
    acc_scores = cross_val_score(lr, Xf, y_tr, cv=cv_obj, scoring="accuracy")
    auc_scores = cross_val_score(lr, Xf, y_tr, cv=cv_obj, scoring="roc_auc_ovr_weighted")

    # McFadden pseudo-R² (log-loss based)
    from sklearn.metrics import log_loss
    lr_full = LogisticRegression(max_iter=500, random_state=42, multi_class="multinomial",
                                 solver="lbfgs")
    lr_full.fit(Xf, y_tr)
    ll_full = -log_loss(y_tr, lr_full.predict_proba(Xf), normalize=False)

    lr_null = LogisticRegression(max_iter=1, random_state=42, fit_intercept=True)
    # Null model = predict class proportions
    classes, counts = np.unique(y_tr, return_counts=True)
    null_proba = np.tile(counts / counts.sum(), (len(y_tr), 1))
    ll_null = -log_loss(y_tr, null_proba, normalize=False)
    mcfadden = 1 - (ll_full / ll_null) if ll_null != 0 else np.nan

    return {
        "acc_mean":  round(np.mean(acc_scores), 4),
        "acc_std":   round(np.std(acc_scores), 4),
        "auc_mean":  round(np.mean(auc_scores), 4),
        "mcfadden":  round(mcfadden, 4),
        "n_features": len(features),
        "features":  features,
    }


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("8_combtest", "Compare feature combinations — find the minimal set that maximises performance.")

    if not st.session_state.get("factor_done"):
        st.warning("Complete **Stage 7 · Factor Analysis** first.")
        return

    X_train = st.session_state["X_train"].copy()
    X_test  = st.session_state["X_test"].copy()
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    all_features = list(X_train.columns)
    st.markdown(f"**{len(all_features)} features** available.")
    st.caption(
        "Proxy model: Logistic Regression (fast). "
        "Scores guide selection — final model is the PyTorch NN in Stage 9."
    )

    if "comb_results" not in st.session_state:
        st.session_state["comb_results"] = []

    st.divider()

    # ── Mode tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Forward Stepwise", "Backward Stepwise", "Manual Explorer"])

    # ── Forward stepwise ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Forward Stepwise Selection")
        st.caption("Starts with no features; at each step adds the feature that most improves CV accuracy.")

        fw_metric = st.selectbox("Optimise by", ["acc_mean", "auc_mean", "mcfadden"],
                                 key="fw_metric")
        fw_cv     = st.slider("CV folds", 3, 10, 5, key="fw_cv")
        fw_max    = st.slider("Max features to add", 1, len(all_features),
                              min(15, len(all_features)), key="fw_max")

        if st.button("Run forward stepwise", type="primary", key="run_fw"):
            selected = []
            remaining = all_features.copy()
            fw_history = []
            bar = st.progress(0)
            status_txt = st.empty()

            for step in range(fw_max):
                best_score = -np.inf
                best_feat  = None
                for feat in remaining:
                    candidate = selected + [feat]
                    s = _score_features(X_train, y_train, candidate, cv=fw_cv)
                    if s[fw_metric] > best_score:
                        best_score = s[fw_metric]
                        best_feat  = feat
                        best_s     = s

                if best_feat is None:
                    break
                selected.append(best_feat)
                remaining.remove(best_feat)
                fw_history.append({**best_s, "step": step+1, "added": best_feat})
                bar.progress((step+1) / fw_max)
                status_txt.caption(f"Step {step+1}: added **{best_feat}** ({fw_metric}={best_score:.4f})")

            bar.empty(); status_txt.empty()
            st.session_state["fw_history"] = fw_history
            st.success(f"Forward stepwise complete — {len(selected)} features.")

        if st.session_state.get("fw_history"):
            hist = pd.DataFrame(st.session_state["fw_history"])
            fw_fig = px.line(hist, x="step", y=fw_metric, markers=True,
                             title="Forward stepwise — score by step",
                             labels={"step": "Step (features added)", fw_metric: fw_metric})
            fw_fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fw_fig, use_container_width=True)
            st.dataframe(hist[["step","added","acc_mean","auc_mean","mcfadden","n_features"]],
                         use_container_width=True, hide_index=True)
            best_row = hist.loc[hist[fw_metric].idxmax()]
            if st.button(f"Use forward result ({int(best_row['n_features'])} features)",
                         key="use_fw"):
                st.session_state["selected_features"] = best_row["features"]
                st.success("Features set for Stage 9. You can still override in Manual Explorer.")

    # ── Backward stepwise ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Backward Stepwise Elimination")
        st.caption("Starts with all features; at each step removes the least important one.")

        bw_metric = st.selectbox("Optimise by", ["acc_mean", "auc_mean", "mcfadden"],
                                 key="bw_metric")
        bw_cv     = st.slider("CV folds", 3, 10, 5, key="bw_cv")
        bw_min    = st.slider("Min features to retain", 1, len(all_features) - 1,
                              max(2, len(all_features) - 10), key="bw_min")

        if st.button("Run backward stepwise", type="primary", key="run_bw"):
            remaining = all_features.copy()
            bw_history = []
            bar = st.progress(0)
            status_txt = st.empty()
            total_steps = len(all_features) - bw_min

            step = 0
            while len(remaining) > bw_min:
                best_score = -np.inf
                worst_feat = None
                for feat in remaining:
                    candidate = [f for f in remaining if f != feat]
                    s = _score_features(X_train, y_train, candidate, cv=bw_cv)
                    if s[bw_metric] > best_score:
                        best_score = s[bw_metric]
                        worst_feat = feat
                        best_s     = s

                if worst_feat is None:
                    break
                remaining.remove(worst_feat)
                step += 1
                bw_history.append({**best_s, "step": step, "removed": worst_feat})
                bar.progress(min(step / max(total_steps, 1), 1.0))
                status_txt.caption(f"Step {step}: removed **{worst_feat}** ({bw_metric}={best_score:.4f})")

            bar.empty(); status_txt.empty()
            st.session_state["bw_history"] = bw_history
            st.success(f"Backward stepwise complete — {len(remaining)} features remain.")

        if st.session_state.get("bw_history"):
            hist = pd.DataFrame(st.session_state["bw_history"])
            bw_fig = px.line(hist, x="n_features", y=bw_metric, markers=True,
                             title="Backward stepwise — score vs. feature count",
                             labels={"n_features": "Features remaining", bw_metric: bw_metric})
            bw_fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(bw_fig, use_container_width=True)
            st.dataframe(hist[["step","removed","acc_mean","auc_mean","mcfadden","n_features"]],
                         use_container_width=True, hide_index=True)
            best_row = hist.loc[hist[bw_metric].idxmax()]
            if st.button(f"Use backward result ({int(best_row['n_features'])} features)",
                         key="use_bw"):
                st.session_state["selected_features"] = best_row["features"]
                st.success("Features set for Stage 9.")

    # ── Manual explorer ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Manual Feature Explorer")
        st.caption("Pick any combination and score it instantly.")

        default_sel = st.session_state.get("selected_features", all_features[:10])
        manual_sel  = st.multiselect(
            "Select features to test:",
            options=all_features,
            default=[f for f in default_sel if f in all_features],
            key="manual_feat_sel",
        )

        man_cv = st.slider("CV folds", 3, 10, 5, key="man_cv")

        if st.button("Score this combination", key="score_manual"):
            if not manual_sel:
                st.warning("Select at least one feature.")
            else:
                with st.spinner("Scoring..."):
                    s = _score_features(X_train, y_train, manual_sel, cv=man_cv)
                st.session_state["comb_results"].append({
                    **s,
                    "label": f"Manual ({len(manual_sel)} feat)",
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                })
                st.success(f"Accuracy: {s['acc_mean']:.4f} ± {s['acc_std']:.4f}  |  "
                           f"AUC: {s['auc_mean']:.4f}  |  McFadden R²: {s['mcfadden']:.4f}")

        if st.button("Use this combination for Stage 9", key="use_manual"):
            st.session_state["selected_features"] = manual_sel
            st.success(f"{len(manual_sel)} features selected for the neural network.")

    st.divider()

    # ── Results comparison table ──────────────────────────────────────────────
    st.markdown("### All scored combinations")
    results = st.session_state.get("comb_results", [])
    if results:
        res_df = pd.DataFrame([{
            "Label": r.get("label",""),
            "N features": r["n_features"],
            "CV Accuracy": r["acc_mean"],
            "CV AUC": r["auc_mean"],
            "McFadden R²": r["mcfadden"],
        } for r in results])
        st.dataframe(res_df.sort_values("CV Accuracy", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No combinations scored yet.")

    st.divider()

    # ── Finalise selection ────────────────────────────────────────────────────
    st.markdown("### Finalise feature set for Stage 9")
    current_sel = st.session_state.get("selected_features", [])
    if current_sel:
        st.success(f"**{len(current_sel)} features** selected: {', '.join(current_sel[:10])}"
                   f"{'…' if len(current_sel)>10 else ''}")
        if st.button("Clear selection", key="clear_sel"):
            st.session_state["selected_features"] = None
            st.rerun()
    else:
        st.info("No feature set finalised yet. Use one of the stepwise methods or manual explorer above.")
