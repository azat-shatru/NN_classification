"""
Stage 9 — PyTorch Neural Network
Multiclass classification (4 classes) with:
  - 2 hidden layers (hidden1 = 2-4× input, hidden2 = hidden1//2)
  - ReLU activations, Dropout=0.3
  - nn.CrossEntropyLoss (includes softmax — no explicit Softmax in forward())
  - Adam optimizer, ReduceLROnPlateau scheduler
  - 5-Fold Stratified Cross-Validation
  - Early stopping
  - SHAP values for variable importance
  - Confusion matrix, AUC (OVR), Accuracy, McFadden pseudo-R²
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              log_loss, classification_report)
import io, time

from utils.state import init
from utils.ui import stage_header


# ── check torch ───────────────────────────────────────────────────────────────

def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


# ── model definition ──────────────────────────────────────────────────────────

def _build_model(input_size: int, hidden1: int, hidden2: int, n_classes: int,
                 dropout: float):
    import torch
    import torch.nn as nn

    class TabularNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, n_classes),   # raw logits — CrossEntropyLoss handles softmax
            )

        def forward(self, x):
            return self.net(x)

    return TabularNN()


# ── training loop ─────────────────────────────────────────────────────────────

def _train_fold(X_tr_np, y_tr_np, X_val_np, y_val_np,
                input_size, hidden1, hidden2, n_classes, dropout,
                lr, weight_decay, n_epochs, patience, batch_size):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cpu")
    model  = _build_model(input_size, hidden1, hidden2, n_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)

    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.long)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    best_val_loss = np.inf
    best_state    = None
    no_improve    = 0
    history       = []

    for epoch in range(n_epochs):
        # ── train ──
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(Xb)
        ep_loss /= len(X_tr_np)

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1).numpy()
            val_acc    = accuracy_score(y_val_np, val_preds)

        history.append({"epoch": epoch+1, "train_loss": ep_loss,
                        "val_loss": val_loss, "val_acc": val_acc})

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ── metrics helpers ───────────────────────────────────────────────────────────

def _predict(model, X_np):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np, dtype=torch.float32))
        proba  = torch.softmax(logits, dim=1).numpy()
        preds  = np.argmax(proba, axis=1)
    return preds, proba


def _mcfadden(y_true, proba):
    """McFadden pseudo-R² = 1 - LL_model / LL_null."""
    ll_model = -log_loss(y_true, proba, normalize=False)
    classes, counts = np.unique(y_true, return_counts=True)
    null_proba = np.tile(counts / counts.sum(), (len(y_true), 1))
    ll_null = -log_loss(y_true, null_proba, normalize=False)
    return 1 - (ll_model / ll_null) if ll_null != 0 else np.nan


# ── main page ─────────────────────────────────────────────────────────────────

def show():
    init()
    stage_header("9_nn", "Train the PyTorch multiclass neural network and evaluate performance.")

    if not _check_torch():
        st.error("`torch` not installed. Run: `pip install torch` then restart.")
        return

    if st.session_state.get("selected_features") is None:
        st.warning("Finalise a feature set in **Stage 8 · Combination Testing** first.")
        return

    import torch

    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]
    features = st.session_state["selected_features"]

    # Filter to selected features
    avail = [f for f in features if f in X_train.columns]
    if len(avail) < len(features):
        st.warning(f"{len(features)-len(avail)} selected features not found in dataset — skipped.")
    features = avail

    X_tr_np = X_train[features].values.astype(np.float32)
    X_te_np = X_test[features].values.astype(np.float32)
    y_tr_np = y_train.values.astype(np.int64)
    y_te_np = y_test.values.astype(np.int64)

    # Label-encode y if not already integer 0-based
    classes = sorted(np.unique(y_tr_np))
    cls_map = {c: i for i, c in enumerate(classes)}
    y_tr_np = np.array([cls_map[v] for v in y_tr_np])
    y_te_np = np.array([cls_map.get(v, -1) for v in y_te_np])
    n_classes  = len(classes)
    input_size = len(features)

    st.markdown(f"**Input features:** {input_size}  |  **Classes:** {n_classes}  |  "
                f"**Train rows:** {len(X_tr_np)}  |  **Test rows:** {len(X_te_np)}")

    st.divider()

    # ── Architecture config ───────────────────────────────────────────────────
    st.markdown("### Architecture")
    ac1, ac2, ac3, ac4 = st.columns(4)
    h1_mult = ac1.slider("Hidden 1 size (× input)", 1, 4, 2,
                         help="hidden1 = multiplier × input_size")
    h1 = max(8, h1_mult * input_size)
    h2 = max(4, h1 // 2)
    ac2.metric("Hidden 1 neurons", h1)
    ac3.metric("Hidden 2 neurons", h2)
    dropout = ac4.slider("Dropout rate", 0.0, 0.6, 0.3, 0.05)

    st.caption(f"Architecture: {input_size} → {h1} (ReLU, Dropout) → {h2} (ReLU, Dropout) → {n_classes} logits")

    st.divider()

    # ── Training config ───────────────────────────────────────────────────────
    st.markdown("### Training settings")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    lr           = tc1.number_input("Learning rate",  1e-5, 1e-1, 1e-3, format="%.5f")
    weight_decay = tc2.number_input("Weight decay (L2)", 0.0, 0.1, 1e-4, format="%.5f")
    n_epochs     = tc3.number_input("Max epochs", 50, 2000, 500, 50)
    patience     = tc4.number_input("Early stop patience", 5, 100, 20, 5)
    batch_size   = tc5.selectbox("Batch size", [16, 32, 64, 128, 256], index=2)

    n_folds = st.slider("Cross-validation folds", 3, 10, 5, key="nn_cv_folds")

    st.divider()

    # ── Train ─────────────────────────────────────────────────────────────────
    st.markdown("### Train")

    if st.button("Train Neural Network", type="primary", key="train_nn"):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        fold_histories = []
        best_model = None
        best_val_acc = -1

        progress = st.progress(0)
        status   = st.empty()

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tr_np, y_tr_np)):
            status.caption(f"Training fold {fold_i+1} / {n_folds} …")
            Xf_tr, Xf_val = X_tr_np[tr_idx], X_tr_np[val_idx]
            yf_tr, yf_val = y_tr_np[tr_idx], y_tr_np[val_idx]

            model, hist = _train_fold(
                Xf_tr, yf_tr, Xf_val, yf_val,
                input_size=input_size, hidden1=h1, hidden2=h2,
                n_classes=n_classes, dropout=dropout,
                lr=lr, weight_decay=weight_decay,
                n_epochs=int(n_epochs), patience=int(patience),
                batch_size=int(batch_size),
            )
            val_preds, val_proba = _predict(model, Xf_val)
            fold_acc = accuracy_score(yf_val, val_preds)
            fold_auc = roc_auc_score(yf_val, val_proba, multi_class="ovr",
                                     average="weighted")
            fold_r2  = _mcfadden(yf_val, val_proba)

            fold_metrics.append({"fold": fold_i+1, "val_acc": fold_acc,
                                  "val_auc": fold_auc, "mcfadden_r2": fold_r2,
                                  "epochs": len(hist)})
            fold_histories.append(hist.assign(fold=fold_i+1))

            if fold_acc > best_val_acc:
                best_val_acc = fold_acc
                best_model   = model

            progress.progress((fold_i+1) / n_folds)

        progress.empty(); status.empty()

        # ── Final model on full train set ──────────────────────────────────
        status2 = st.empty()
        status2.caption("Fitting final model on full training set …")
        final_model, final_hist = _train_fold(
            X_tr_np, y_tr_np, X_tr_np, y_tr_np,
            input_size=input_size, hidden1=h1, hidden2=h2,
            n_classes=n_classes, dropout=dropout,
            lr=lr, weight_decay=weight_decay,
            n_epochs=int(n_epochs), patience=int(patience),
            batch_size=int(batch_size),
        )
        status2.empty()

        st.session_state["nn_model"]          = final_model
        st.session_state["nn_fold_metrics"]   = fold_metrics
        st.session_state["nn_fold_histories"] = fold_histories
        st.session_state["nn_input_features"] = features
        st.session_state["nn_trained"]        = True
        st.session_state["nn_classes"]        = classes
        st.success("Training complete!")

    if not st.session_state.get("nn_trained"):
        return

    # ── Results ───────────────────────────────────────────────────────────────
    final_model = st.session_state["nn_model"]
    fold_metrics = st.session_state["nn_fold_metrics"]
    fold_histories = st.session_state["nn_fold_histories"]
    classes_out  = st.session_state["nn_classes"]

    st.divider()
    st.markdown("### Cross-Validation Results")

    metrics_df = pd.DataFrame(fold_metrics)
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Mean CV Accuracy", f"{metrics_df['val_acc'].mean():.4f}",
               delta=f"±{metrics_df['val_acc'].std():.4f}")
    cm2.metric("Mean CV AUC (OVR)", f"{metrics_df['val_auc'].mean():.4f}",
               delta=f"±{metrics_df['val_auc'].std():.4f}")
    cm3.metric("Mean McFadden R²",   f"{metrics_df['mcfadden_r2'].mean():.4f}")
    cm4.metric("Avg Epochs",         f"{metrics_df['epochs'].mean():.0f}")

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Learning curves ────────────────────────────────────────────────────────
    st.markdown("#### Learning curves (fold 1)")
    hist1 = fold_histories[0]
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(x=hist1["epoch"], y=hist1["train_loss"],
                                name="Train loss", mode="lines"))
    fig_lc.add_trace(go.Scatter(x=hist1["epoch"], y=hist1["val_loss"],
                                name="Val loss", mode="lines"))
    fig_lc.update_layout(title="Loss curves (fold 1)",
                         xaxis_title="Epoch", yaxis_title="Loss",
                         height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig_lc, use_container_width=True)

    st.divider()

    # ── Test set evaluation ────────────────────────────────────────────────────
    st.markdown("### Test Set Evaluation")
    te_preds, te_proba = _predict(final_model, X_te_np)
    valid_test = y_te_np >= 0   # exclude any unseen classes
    te_acc  = accuracy_score(y_te_np[valid_test], te_preds[valid_test])
    te_auc  = roc_auc_score(y_te_np[valid_test], te_proba[valid_test],
                            multi_class="ovr", average="weighted")
    te_r2   = _mcfadden(y_te_np[valid_test], te_proba[valid_test])

    t1, t2, t3 = st.columns(3)
    t1.metric("Test Accuracy",    f"{te_acc:.4f}")
    t2.metric("Test AUC (OVR)",   f"{te_auc:.4f}")
    t3.metric("Test McFadden R²", f"{te_r2:.4f}")

    # Classification report
    cls_labels = [str(c) for c in classes_out]
    report_txt = classification_report(
        y_te_np[valid_test], te_preds[valid_test], target_names=cls_labels
    )
    with st.expander("Full classification report (test set)"):
        st.code(report_txt)

    # Confusion matrix
    cm_arr = confusion_matrix(y_te_np[valid_test], te_preds[valid_test])
    cm_fig = px.imshow(
        cm_arr, text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=cls_labels, y=cls_labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix (test set)",
    )
    cm_fig.update_layout(height=350, margin=dict(t=50, b=20))
    st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()

    # ── SHAP values ────────────────────────────────────────────────────────────
    st.markdown("### Feature Importance (SHAP)")
    if st.button("Compute SHAP values", key="compute_shap"):
        try:
            import shap
            import torch

            final_model.eval()
            background = torch.tensor(X_tr_np[:min(100, len(X_tr_np))], dtype=torch.float32)

            def model_fn(x_np):
                with torch.no_grad():
                    logits = final_model(torch.tensor(x_np, dtype=torch.float32))
                    return torch.softmax(logits, dim=1).numpy()

            explainer  = shap.KernelExplainer(model_fn, background.numpy())
            shap_vals  = explainer.shap_values(X_te_np[:min(50, len(X_te_np))],
                                               nsamples=100)

            # Mean |SHAP| per feature across all classes
            if isinstance(shap_vals, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            else:
                mean_shap = np.abs(shap_vals).mean(axis=0)

            shap_df = pd.DataFrame({
                "Feature": features,
                "Mean |SHAP|": mean_shap,
            }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

            st.session_state["shap_df"] = shap_df
            st.success("SHAP values computed.")
        except ImportError:
            st.error("`shap` not installed. Run: `pip install shap` then restart.")
        except Exception as e:
            st.error(f"SHAP error: {e}")

    if st.session_state.get("shap_df") is not None:
        shap_df = st.session_state["shap_df"]
        shap_fig = px.bar(
            shap_df, x="Mean |SHAP|", y="Feature",
            orientation="h",
            color="Mean |SHAP|", color_continuous_scale="Blues",
            title="SHAP Feature Importance (mean |SHAP value|)",
        )
        shap_fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(350, len(shap_df) * 22 + 80),
            margin=dict(t=50, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Re-train button ────────────────────────────────────────────────────────
    if st.button("Re-train with different settings", key="retrain_nn"):
        st.session_state["nn_trained"] = False
        st.session_state["shap_df"]   = None
        st.rerun()
