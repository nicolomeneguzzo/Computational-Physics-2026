"""Traditional ML + neural-network trainers."""

import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier #nicolò
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier #aggiunta sara
from sklearn.model_selection import RandomizedSearchCV  #aggiunta enrica 
from .models.trees import SimpleRandomForest, SimpleExtraTrees   #aggiunta enrica
from sklearn.linear_model import LogisticRegression #nicolò
from sklearn.calibration import CalibratedClassifierCV #nicolò


from .models.network import SimpleNN


# Model factory 
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
        'XGBoost':        XGBClassifier(
                              tree_method='gpu_hist' if use_gpu else 'hist'),
    }



#  Metrics helper 
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



#  Traditional ML 
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
                f"  [{m['dataset']}] Acc={m['accuracy']:.4f}%  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f }"
            )

        gc.collect()

    return models




def train_voting(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    models:  dict | None = None,
    voting: str | None = None, 
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
    ('svc', models['Linear SVC']),
    ('rf',  models['Random Forest']),
    ('et',  models['Extra Trees']),
    ('xgb', models['XGBoost']),
    ('lgbm', models['LightGBM']),
]
    voting_clf = _Voting(estimators=estimators, voting=voting)
    voting_clf.fit(X_train, y_train)
    print("Voting Classifier trained.")

    train_m = compute_metrics(y_train, voting_clf.predict(X_train), 'Training',   'Voting Classifier')
    val_m   = compute_metrics(y_val,   voting_clf.predict(X_val),   'Validation', 'Voting Classifier')
    for m in (train_m, val_m):
        print(
            f"  [{m['dataset']}] Acc={m['accuracy']:.4f}%  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
        )

    return voting_clf



#  Neural Network 
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

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    # Training loop
    for epoch in range(num_epochs):

        #  TRAIN 
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

        #  TRAIN METRICS 
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # VALIDATION 
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

        #SAVE HISTORY 
        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        # LOGGING 
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.2f}% | "
            f"val_acc={val_acc:.2f}%"
        )

    # Final metrics
    final_metrics = {
        "final_loss": history["train_loss"][-1],
        "final_train_acc": history["train_accuracy"][-1],
        "final_val_acc": history["val_accuracy"][-1],
    }

    return model, history, final_metrics





def train_trees_with_tuning(X_train, y_train, X_val, y_val, n_iter=10, cv=5):
    """RF e ET con hyperparameter tuning via RandomizedSearchCV."""
    param_dist_rf= {
        'n_estimators':      [50, 100, 150, 200, 300],
        'max_depth':         [5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['log2', 0.5, 'sqrt', None],
        'max_samples':       [0.6, 0.7, 0.8] #per overfitting 
    
    }
    param_dist_et = {
        'n_estimators':      [50, 100, 150, 200, 300],
        'max_depth':         [5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2', 0.5]
    }
    models = {}
    for name, base, param_dist in [('Random Forest', SimpleRandomForest(), param_dist_rf),
                       ('Extra Trees',   SimpleExtraTrees(), param_dist_et)]:
        tuner = RandomizedSearchCV(base, param_dist, n_iter=n_iter,
                                   cv=cv, scoring='f1_macro', n_jobs=3)
        tuner.fit(X_train, y_train)
        models[name] = tuner.best_estimator_
        print(f"{name} best params: {tuner.best_params_}")
        for X, y, split in [(X_train, y_train, 'Train'), (X_val, y_val, 'Val')]:
            m = compute_metrics(y, models[name].predict(X), split, name)
            print(f"Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
            f"\nConfusion Matrix:\n{m['confusion_matrix']}\n")
    return models




def tune_model(
    model,
    param_grid: dict,
    X_train,
    y_train,
    *,
    scoring: str = "f1_macro",
    cv: int = 3,
    n_iter: int = 10,
    n_jobs: int = 3,
    random_state: int = 42,
    verbose: int = 0,
):
    """
    Generic hyperparameter tuning function.

    Parameters
    ----------
    model : estimator
        Any sklearn-compatible model.
    param_grid : dict
        Hyperparameter search space.
    X_train, y_train :
        Training data.
    scoring : str
        Optimization metric.
    cv : int
        Cross-validation folds.
    n_iter : int
        Number of sampled parameter combinations.
    n_jobs : int
        Parallel jobs.
    random_state : int
        Random seed.
    verbose : int
        Verbosity level.

    Returns
    -------
    best_model :
        Best fitted estimator.
    best_params : dict
        Best hyperparameters.
    best_score : float
        Best cross-validation score.
    """

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
    )




def evaluate_single_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """
    Train (already fitted model assumed) + compute all metrics on train/val.
    """

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)

    train_metrics = compute_metrics(y_train, train_pred, "Training", model_name)
    val_metrics   = compute_metrics(y_val, val_pred, "Validation", model_name)

    for m in (train_metrics, val_metrics):
        print(
            f"[{m['dataset']}] Acc={m['accuracy']:.4f}% "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
            f"\nConfusion Matrix:\n{m['confusion_matrix']}\n"
        )

    return train_metrics, val_metrics




def make_stacking_classifier(
    models: dict,
    n_jobs: int = -1,
    final_estimator=None,
    passthrough: bool = False,
    cv: int = 10,  #from the gridsearch
    stack_method: str = 'predict_proba',
    ) -> StackingClassifier:

    # Default meta-model
    if final_estimator is None:
        final_estimator = LogisticRegression(
            max_iter=1000
        )

    estimators = [
        ('svc',  models['Linear SVC']),
        ('rf',   models['Random Forest']),
        ('et',   models['Extra Trees']),
        ('xgb',  models['XGBoost']),
        ('lgbm', models['LightGBM']),
    ]


    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=passthrough,
        stack_method=stack_method,
        cv=cv,
        n_jobs=n_jobs,
    )




def train_stacking(
    X_train, y_train,
    X_val, y_val,
    models: dict,            
    n_jobs: int = -1,
    final_estimator=None,
    passthrough: bool = False,
    cv: int = 10,              # optimal value founded from GridSearch
    stack_method: str = 'predict_proba',
) -> StackingClassifier:
    stacking_clf = make_stacking_classifier(
        models=models,
        n_jobs=n_jobs,
        final_estimator=final_estimator,
        passthrough=passthrough,
        cv=cv,
        stack_method=stack_method,
    )
    print("Starting stacking training...")
    stacking_clf.fit(X_train, y_train)
    print("Stacking Classifier trained.")

    train_m = compute_metrics(y_train, stacking_clf.predict(X_train), 'Training', 'Stacking Classifier')
    val_m = compute_metrics(y_val, stacking_clf.predict(X_val), 'Validation', 'Stacking Classifier')
    for m in (train_m, val_m):
        print(f"  [{m['dataset']}] Acc={m['accuracy']:.4f}%  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    return stacking_clf