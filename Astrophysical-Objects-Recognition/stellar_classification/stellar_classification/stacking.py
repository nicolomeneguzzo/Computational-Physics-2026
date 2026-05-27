import numpy as np
import torch
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
#base models
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

#for making calibrated probability
from sklearn.calibration import CalibratedClassifierCV

#meta_learner
from sklearn.linear_model import LogisticRegression

from .trainer import _make_models, train_traditional, compute_metrics
from .models.network import SimpleNN



def _make_stacking_models() -> dict:
    use_gpu = torch.cuda.is_available()
    return {
        'Linear SVC':    LinearSVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'CatBoost':      CatBoostClassifier(task_type='GPU' if use_gpu else 'CPU', verbose=0),
        'LightGBM':      LGBMClassifier(device='gpu' if use_gpu else 'cpu'),
    }

def stack_training_models(
    X_train: np.ndarray, y_train: np.ndarray,
) -> dict:
    models = _make_stacking_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained.")
    return models 
    

def make_stacking_classifier(
    models: dict,
    n_jobs: int = -1,
    final_estimator=None,
    passthrough: bool = False,
    cv: int = 5,
    stack_method: str = 'predict_proba',
) -> StackingClassifier:

    # Default meta-model
    if final_estimator is None:
        final_estimator = LogisticRegression(
            max_iter=1000
        )

    # Calibrated SVC for predict_proba
    calibrated_svc = CalibratedClassifierCV(
        models['Linear SVC'],
        cv=None
    )

    estimators = [
        ('svc', calibrated_svc),
        ('dt', models['Decision Tree']),
        ('rf', models['Random Forest']),
        ('catboost', models['CatBoost']),
        ('lgbm', models['LightGBM']),
    ]

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=passthrough,
        stack_method=stack_method,
        cv=cv,
        n_jobs=n_jobs,
    )

    return stacking_clf



def train_stacking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = -1,
    models: dict | None = None,

    final_estimator=None,
    passthrough: bool = False,
    cv: int = 5,
    stack_method: str = 'predict_proba',
) -> StackingClassifier:

    # Train base models if needed
    if models is None:
        models = stack_training_models(X_train, y_train)

    # Build classifier
    stacking_clf = make_stacking_classifier(
        models=models,
        n_jobs=n_jobs,
        final_estimator=final_estimator,
        passthrough=passthrough,
        cv=cv,
        stack_method=stack_method,
    )

    # Train
    print("Starting stacking training...")
    stacking_clf.fit(X_train, y_train)
    print("Stacking Classifier trained.")

    # Metrics
    train_m = compute_metrics(
        y_train,
        stacking_clf.predict(X_train),
        'Training',
        'Stacking Classifier'
    )

    val_m = compute_metrics(
        y_val,
        stacking_clf.predict(X_val),
        'Validation',
        'Stacking Classifier'
    )

    for m in (train_m, val_m):
        print(
            f"  [{m['dataset']}] "
            f"Acc={m['accuracy']:.2f}%  "
            f"P={m['precision']:.2f}  "
            f"R={m['recall']:.2f}  "
            f"F1={m['f1']:.2f}"
        )

    return stacking_clf

