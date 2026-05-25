"""Traditional ML + neural-network trainers."""

import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from .models.network import SimpleNN


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_models() -> dict:
    """Return a fresh dict of unfitted estimators."""
    use_gpu = torch.cuda.is_available()
    return {
        'Linear SVC':     LinearSVC(),
        'Decision Tree':  DecisionTreeClassifier(),
        'Random Forest':  RandomForestClassifier(),
        'CatBoost':       CatBoostClassifier(
                              task_type='GPU' if use_gpu else 'CPU', verbose=0),
        'LightGBM':       LGBMClassifier(
                              device='gpu' if use_gpu else 'cpu'),
    }


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, dataset_name: str, model_name: str) -> dict:
    """Compute classification metrics and return as a dict.

    ``accuracy`` is stored as a percentage (0-100) for display convenience.
    """
    return {
        'model':            model_name,
        'dataset':          dataset_name,
        'accuracy':         accuracy_score(y_true, y_pred) * 100,
        'precision':        precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':           recall_score(y_true, y_pred,    average='macro', zero_division=0),
        'f1':               f1_score(y_true, y_pred,        average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


# ── Traditional ML ────────────────────────────────────────────────────────────

def train_traditional(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> dict:
    """Fit all traditional ML models and return a dict of fitted estimators.

    Parameters
    ----------
    X_train, y_train : arrays
        SMOTE-augmented training data.
    X_val, y_val : arrays
        Validation data (used only for printing per-model metrics).

    Returns
    -------
    models : dict[str, fitted estimator]
    """
    models = _make_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")

        train_metrics = compute_metrics(y_train, model.predict(X_train), 'Training',   name)
        val_metrics   = compute_metrics(y_val,   model.predict(X_val),   'Validation', name)

        for m in (train_metrics, val_metrics):
            print(
                f"  [{m['dataset']}] Acc={m['accuracy']:.2f}%  "
                f"P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}"
            )

        gc.collect()

    return models


def train_voting(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    models:  dict | None = None,
) -> VotingClassifier:
    """Build and fit a hard VotingClassifier over all traditional models.

    Parameters
    ----------
    models : dict, optional
        Pre-fitted estimators from :func:`train_traditional`.  If *None* the
        models are trained internally (useful if you want the voting classifier
        only).
    """
    if models is None:
        models = train_traditional(X_train, y_train, X_val, y_val)

    # Custom subclass that fixes prediction shape for hard voting
    class _Voting(VotingClassifier):
        def _predict(self, X):
            preds = [est.predict(X).ravel() for est in self.estimators_]
            return np.asarray(preds).T

    estimators = [
        ('svc',      models['Linear SVC']),
        ('dt',       models['Decision Tree']),
        ('rf',       models['Random Forest']),
        ('catboost', models['CatBoost']),
        ('lgbm',     models['LightGBM']),
    ]
    voting_clf = _Voting(estimators=estimators, voting='hard')
    voting_clf.fit(X_train, y_train)
    print("Voting Classifier trained.")

    train_m = compute_metrics(y_train, voting_clf.predict(X_train), 'Training',   'Voting Classifier')
    val_m   = compute_metrics(y_val,   voting_clf.predict(X_val),   'Validation', 'Voting Classifier')
    for m in (train_m, val_m):
        print(
            f"  [{m['dataset']}] Acc={m['accuracy']:.2f}%  "
            f"P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}"
        )

    return voting_clf


def train_neural(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    lr: float = 0.001,
    num_epochs: int = 10,
):
    """Train a given PyTorch model and return training history."""

    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ────────────────────────────────────────────────────────────────────────
    # Training history
    # ────────────────────────────────────────────────────────────────────────

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    # ────────────────────────────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────────────────────────────

    for epoch in range(num_epochs):

        # ── TRAIN ───────────────────────────────────────────────────────────
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # predictions
            _, pred = outputs.max(1)

            correct_train += (pred == y_batch).sum().item()
            total_train += y_batch.size(0)

        # ── TRAIN METRICS ───────────────────────────────────────────────────
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # ── VALIDATION ──────────────────────────────────────────────────────
        model.eval()

        correct_val = 0
        total_val = 0

        with torch.no_grad():

            for Xb, yb in val_loader:

                Xb = Xb.to(device)
                yb = yb.to(device)

                outputs = model(Xb)

                _, pred = outputs.max(1)

                correct_val += (pred == yb).sum().item()
                total_val += yb.size(0)

        val_acc = 100 * correct_val / total_val

        # ── SAVE HISTORY ────────────────────────────────────────────────────
        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        # ── LOGGING ─────────────────────────────────────────────────────────
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.2f}% | "
            f"val_acc={val_acc:.2f}%"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Final metrics
    # ────────────────────────────────────────────────────────────────────────

    final_metrics = {
        "final_loss": history["train_loss"][-1],
        "final_train_acc": history["train_accuracy"][-1],
        "final_val_acc": history["val_accuracy"][-1],
    }

    return model, history, final_metrics
