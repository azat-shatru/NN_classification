"""
Stage 9 — PyTorch Neural Network
Architecture sliders update caption immediately.
Training runs with dcc.Loading wrapper.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              log_loss, classification_report)

from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server_store


def _check_torch():
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False


def _build_model(input_size, hidden1, hidden2, n_classes, dropout):
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
                nn.Linear(hidden2, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    return TabularNN()


def _train_fold(X_tr_np, y_tr_np, X_val_np, y_val_np,
                input_size, hidden1, hidden2, n_classes, dropout,
                lr, weight_decay, n_epochs, patience, batch_size):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cpu")
    model = _build_model(input_size, hidden1, hidden2, n_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)

    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.long)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    best_val_loss = np.inf
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(Xb)
        ep_loss /= len(X_tr_np)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val_np, val_preds)

        history.append({"epoch": epoch+1, "train_loss": ep_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def _predict(model, X_np):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(proba, axis=1)
    return preds, proba


def _mcfadden(y_true, proba):
    ll_model = -log_loss(y_true, proba, normalize=False)
    classes, counts = np.unique(y_true, return_counts=True)
    null_proba = np.tile(counts / counts.sum(), (len(y_true), 1))
    ll_null = -log_loss(y_true, null_proba, normalize=False)
    return 1 - (ll_model / ll_null) if ll_null != 0 else float("nan")


def layout(state: dict) -> html.Div:
    if not _check_torch():
        return dbc.Alert("`torch` not installed. Run: pip install torch", color="danger")

    if not state.get("selected_features"):
        return dbc.Alert("Finalise a feature set in Stage 8 — Combination Testing first.", color="warning")

    X_train = server_store.get_df("X_train")
    if X_train is None:
        return dbc.Alert("No data found.", color="warning")

    features = state.get("selected_features") or []
    avail = [f for f in features if f in X_train.columns]
    if not avail:
        return dbc.Alert("None of the selected features are available in the current dataset.", color="danger")

    input_size = len(avail)
    X_te_np = server_store.get_df("X_test")
    n_train = len(X_train)
    n_test = len(X_te_np) if X_te_np is not None else 0

    # Check if trained
    nn_trained = state.get("nn_trained", False)
    results_section = html.Div()

    if nn_trained:
        fold_metrics = server_store.get_val("nn_fold_metrics")
        fold_histories = server_store.get_val("nn_fold_histories")
        final_model = server_store.get_val("nn_final_model")
        classes_out = server_store.get_val("nn_classes")

        if fold_metrics:
            metrics_df = pd.DataFrame(fold_metrics)
            m_acc = metrics_df["val_acc"].mean()
            m_auc = metrics_df["val_auc"].mean()
            m_r2 = metrics_df["mcfadden_r2"].mean()
            m_std = metrics_df["val_acc"].std()

            # Learning curve from fold 1
            hist1 = fold_histories[0] if fold_histories else pd.DataFrame()
            lc_fig = go.Figure()
            if not hist1.empty:
                lc_fig.add_trace(go.Scatter(x=hist1["epoch"], y=hist1["train_loss"], name="Train loss"))
                lc_fig.add_trace(go.Scatter(x=hist1["epoch"], y=hist1["val_loss"], name="Val loss"))
                lc_fig.update_layout(title="Loss curves (fold 1)", xaxis_title="Epoch",
                                     yaxis_title="Loss", height=300, margin=dict(t=40, b=20))

            # Test evaluation
            te_results_section = html.Div()
            if final_model is not None and X_te_np is not None:
                y_test = server_store.get_df("y_test")
                if y_test is not None:
                    nn_features = server_store.get_val("nn_input_features") or avail
                    avail_nn = [f for f in nn_features if f in X_te_np.columns]
                    X_te_arr = X_te_np[avail_nn].values.astype(np.float32)
                    y_te_np = np.array(y_test.values)
                    # Re-encode
                    classes_arr = sorted(np.unique(y_te_np))
                    cls_map = {c: i for i, c in enumerate(classes_arr)}
                    y_te_enc = np.array([cls_map.get(v, -1) for v in y_te_np])
                    te_preds, te_proba = _predict(final_model, X_te_arr)
                    valid = y_te_enc >= 0
                    te_acc = accuracy_score(y_te_enc[valid], te_preds[valid])
                    try:
                        te_auc = roc_auc_score(y_te_enc[valid], te_proba[valid], multi_class="ovr", average="weighted")
                    except Exception:
                        te_auc = float("nan")
                    te_r2 = _mcfadden(y_te_enc[valid], te_proba[valid])

                    cm_arr = confusion_matrix(y_te_enc[valid], te_preds[valid])
                    cls_labels = [str(c) for c in classes_arr]
                    cm_fig = px.imshow(
                        cm_arr, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=cls_labels, y=cls_labels,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix (test set)",
                    )
                    cm_fig.update_layout(height=350, margin=dict(t=50, b=20))

                    te_results_section = html.Div([
                        html.H5("Test Set Evaluation"),
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Test Accuracy", className="metric-label"), html.Div(f"{te_acc:.4f}", className="metric-value")]), className="metric-card"), width=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Test AUC (OVR)", className="metric-label"), html.Div(f"{te_auc:.4f}", className="metric-value")]), className="metric-card"), width=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Test McFadden R²", className="metric-label"), html.Div(f"{te_r2:.4f}", className="metric-value")]), className="metric-card"), width=3),
                        ], className="mb-3"),
                        dcc.Graph(figure=cm_fig),
                    ])

            results_section = html.Div([
                html.Hr(className="dash-divider"),
                html.H4("Cross-Validation Results"),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Mean CV Accuracy", className="metric-label"), html.Div(f"{m_acc:.4f} ±{m_std:.4f}", className="metric-value")]), className="metric-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Mean CV AUC", className="metric-label"), html.Div(f"{m_auc:.4f}", className="metric-value")]), className="metric-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Mean McFadden R²", className="metric-label"), html.Div(f"{m_r2:.4f}", className="metric-value")]), className="metric-card"), width=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Avg Epochs", className="metric-label"), html.Div(f"{metrics_df['epochs'].mean():.0f}", className="metric-value")]), className="metric-card"), width=3),
                ], className="mb-3"),
                dbc.Table.from_dataframe(metrics_df, striped=True, hover=True, size="sm"),
                html.H5("Learning curves (fold 1)", className="mt-3"),
                dcc.Graph(figure=lc_fig),
                te_results_section,
                html.Hr(className="dash-divider"),
                html.H5("Feature Importance (SHAP)"),
                dbc.Button("Compute SHAP values", id="s9-shap-btn", color="secondary", n_clicks=0),
                dcc.Loading(html.Div(id="s9-shap-status"), type="circle"),
                html.Div(id="s9-shap-chart"),
            ])

    return html.Div([
        html.Div([
            html.H2("Stage 9 — Neural Network"),
            html.P("Train the PyTorch multiclass neural network and evaluate performance."),
        ], className="stage-header"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Input features", className="metric-label"), html.Div(input_size, className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Train rows", className="metric-label"), html.Div(n_train, className="metric-value")]), className="metric-card"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Div("Test rows", className="metric-label"), html.Div(n_test, className="metric-value")]), className="metric-card"), width=3),
        ], className="mb-3"),

        # Architecture
        dbc.Card(dbc.CardBody([
            html.H5("Architecture"),
            dbc.Row([
                dbc.Col([
                    html.Label("Hidden 1 size (x input)"),
                    dcc.Slider(id="s9-h1-mult", min=1, max=4, step=1, value=2,
                               marks={1:"1x", 2:"2x", 3:"3x", 4:"4x"}),
                ], width=4),
                dbc.Col([
                    html.Label("Dropout rate"),
                    dcc.Slider(id="s9-dropout", min=0.0, max=0.6, step=0.05, value=0.3,
                               marks={0:"0", 0.3:"0.3", 0.6:"0.6"}),
                ], width=4),
            ]),
            html.Div(id="s9-arch-caption", className="mt-2",
                     style={"color": "#6b7280", "fontSize": "0.85rem"}),
        ]), className="mb-3"),

        # Training config
        dbc.Card(dbc.CardBody([
            html.H5("Training settings"),
            dbc.Row([
                dbc.Col([html.Label("Learning rate"), dbc.Input(id="s9-lr", type="number", value=0.001, min=1e-5, max=0.1, step=0.0001)], width=2),
                dbc.Col([html.Label("Weight decay (L2)"), dbc.Input(id="s9-wd", type="number", value=0.0001, min=0.0, max=0.1, step=0.0001)], width=2),
                dbc.Col([html.Label("Max epochs"), dbc.Input(id="s9-epochs", type="number", value=500, min=50, max=2000, step=50)], width=2),
                dbc.Col([html.Label("Early stop patience"), dbc.Input(id="s9-patience", type="number", value=20, min=5, max=100, step=5)], width=2),
                dbc.Col([html.Label("Batch size"), dcc.Dropdown(
                    id="s9-batch",
                    options=[{"label": str(b), "value": b} for b in [16, 32, 64, 128, 256]],
                    value=64, clearable=False,
                )], width=2),
            ]),
            dbc.Row([
                dbc.Col([html.Label("CV folds"), dcc.Slider(id="s9-cv-folds", min=3, max=10, step=1, value=5,
                         marks={3:"3", 5:"5", 10:"10"})], width=6),
            ], className="mt-2"),
        ]), className="mb-3"),

        # Train button
        dbc.Card(dbc.CardBody([
            html.H5("Train"),
            dbc.Button("Train Neural Network", id="s9-train-btn", color="primary", n_clicks=0),
            dcc.Loading(html.Div(id="s9-train-status"), type="circle"),
        ]), className="mb-3"),

        results_section,
    ])


@callback(
    Output("s9-arch-caption", "children"),
    Input("s9-h1-mult", "value"),
    Input("s9-dropout", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def update_arch_caption(h1_mult, dropout, state):
    features = (state or {}).get("selected_features") or []
    X_train = server_store.get_df("X_train")
    if X_train is None:
        return ""
    avail = [f for f in features if f in X_train.columns]
    input_size = len(avail)
    h1 = max(8, (h1_mult or 2) * input_size)
    h2 = max(4, h1 // 2)
    y_train = server_store.get_df("y_train")
    n_classes = len(np.unique(y_train)) if y_train is not None else "?"
    return f"Architecture: {input_size} → {h1} (ReLU, Dropout={dropout}) → {h2} (ReLU, Dropout={dropout}) → {n_classes} logits"


@callback(
    Output("s9-train-status", "children"),
    Output("app-state", "data", allow_duplicate=True),
    Input("s9-train-btn", "n_clicks"),
    State("s9-h1-mult", "value"),
    State("s9-dropout", "value"),
    State("s9-lr", "value"),
    State("s9-wd", "value"),
    State("s9-epochs", "value"),
    State("s9-patience", "value"),
    State("s9-batch", "value"),
    State("s9-cv-folds", "value"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def train_nn(n_clicks, h1_mult, dropout, lr, wd, n_epochs, patience, batch_size, n_folds, state):
    if not n_clicks:
        return no_update, no_update
    if not _check_torch():
        return dbc.Alert("torch not installed.", color="danger"), no_update

    state = dict(state or {})
    features = state.get("selected_features") or []
    X_train = server_store.get_df("X_train")
    X_test = server_store.get_df("X_test")
    y_train = server_store.get_df("y_train")
    y_test = server_store.get_df("y_test")

    if X_train is None or y_train is None:
        return dbc.Alert("No data.", color="danger"), no_update

    avail = [f for f in features if f in X_train.columns]
    if not avail:
        return dbc.Alert("None of selected features found.", color="danger"), no_update

    X_tr_np = X_train[avail].values.astype(np.float32)
    X_te_np = X_test[avail].values.astype(np.float32) if X_test is not None else None
    y_tr_np = y_train.values

    classes = sorted(np.unique(y_tr_np))
    cls_map = {c: i for i, c in enumerate(classes)}
    y_tr_enc = np.array([cls_map[v] for v in y_tr_np], dtype=np.int64)

    n_classes = len(classes)
    input_size = len(avail)
    h1 = max(8, int(h1_mult or 2) * input_size)
    h2 = max(4, h1 // 2)

    kf = StratifiedKFold(n_splits=int(n_folds or 5), shuffle=True, random_state=42)
    fold_metrics = []
    fold_histories = []
    best_model = None
    best_val_acc = -1

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_tr_np, y_tr_enc)):
        Xf_tr, Xf_val = X_tr_np[tr_idx], X_tr_np[val_idx]
        yf_tr, yf_val = y_tr_enc[tr_idx], y_tr_enc[val_idx]
        model, hist = _train_fold(
            Xf_tr, yf_tr, Xf_val, yf_val,
            input_size=input_size, hidden1=h1, hidden2=h2,
            n_classes=n_classes, dropout=float(dropout or 0.3),
            lr=float(lr or 0.001), weight_decay=float(wd or 1e-4),
            n_epochs=int(n_epochs or 500), patience=int(patience or 20),
            batch_size=int(batch_size or 64),
        )
        val_preds, val_proba = _predict(model, Xf_val)
        fold_acc = accuracy_score(yf_val, val_preds)
        try:
            fold_auc = roc_auc_score(yf_val, val_proba, multi_class="ovr", average="weighted")
        except Exception:
            fold_auc = float("nan")
        fold_r2 = _mcfadden(yf_val, val_proba)
        fold_metrics.append({"fold": fold_i+1, "val_acc": fold_acc, "val_auc": fold_auc,
                              "mcfadden_r2": fold_r2, "epochs": len(hist)})
        fold_histories.append(hist.assign(fold=fold_i+1))
        if fold_acc > best_val_acc:
            best_val_acc = fold_acc
            best_model = model

    # Final model on full train
    final_model, _ = _train_fold(
        X_tr_np, y_tr_enc, X_tr_np, y_tr_enc,
        input_size=input_size, hidden1=h1, hidden2=h2,
        n_classes=n_classes, dropout=float(dropout or 0.3),
        lr=float(lr or 0.001), weight_decay=float(wd or 1e-4),
        n_epochs=int(n_epochs or 500), patience=int(patience or 20),
        batch_size=int(batch_size or 64),
    )

    server_store.set_val("nn_final_model", final_model)
    server_store.set_val("nn_fold_metrics", fold_metrics)
    server_store.set_val("nn_fold_histories", [h.to_dict("records") for h in fold_histories] if fold_histories else [])
    server_store.set_val("nn_input_features", avail)
    server_store.set_val("nn_classes", classes)

    # Convert fold_histories to DataFrames for display
    server_store.set_val("nn_fold_histories", fold_histories)

    state["nn_trained"] = True

    m_df = pd.DataFrame(fold_metrics)
    return dbc.Alert(
        f"Training complete! Mean CV Accuracy: {m_df['val_acc'].mean():.4f} ±{m_df['val_acc'].std():.4f}  "
        f"| Mean AUC: {m_df['val_auc'].mean():.4f}. Reload page to see full results.",
        color="success",
    ), state


@callback(
    Output("s9-shap-status", "children"),
    Output("s9-shap-chart", "children"),
    Input("s9-shap-btn", "n_clicks"),
    State("app-state", "data"),
    prevent_initial_call=True,
)
def compute_shap(n_clicks, state):
    if not n_clicks:
        return no_update, no_update
    try:
        import shap
        import torch
    except ImportError as e:
        return dbc.Alert(f"Missing dependency: {e}. Run: pip install shap", color="danger"), html.Div()

    final_model = server_store.get_val("nn_final_model")
    X_train = server_store.get_df("X_train")
    if final_model is None or X_train is None:
        return dbc.Alert("Train the model first.", color="warning"), html.Div()

    features = server_store.get_val("nn_input_features") or list(X_train.columns)
    avail = [f for f in features if f in X_train.columns]
    X_tr_np = X_train[avail].values.astype(np.float32)

    try:
        final_model.eval()
        background = torch.tensor(X_tr_np[:min(100, len(X_tr_np))], dtype=torch.float32)

        def model_fn(x_np):
            with torch.no_grad():
                logits = final_model(torch.tensor(x_np, dtype=torch.float32))
                return torch.softmax(logits, dim=1).numpy()

        explainer = shap.KernelExplainer(model_fn, background.numpy())
        shap_vals = explainer.shap_values(X_tr_np[:min(50, len(X_tr_np))], nsamples=100)

        if isinstance(shap_vals, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_shap = np.abs(shap_vals).mean(axis=0)

        shap_df = pd.DataFrame({
            "Feature": avail,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

        fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                     color="Mean |SHAP|", color_continuous_scale="Blues",
                     title="SHAP Feature Importance")
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          height=max(350, len(shap_df)*22+80),
                          margin=dict(t=50, b=20), coloraxis_showscale=False)

        return dbc.Alert("SHAP values computed.", color="success"), dcc.Graph(figure=fig)
    except Exception as e:
        return dbc.Alert(f"SHAP error: {e}", color="danger"), html.Div()
