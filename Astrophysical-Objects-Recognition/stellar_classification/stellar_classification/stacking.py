import numpy as np
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



def train_stacking(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_jobs,
    models:  dict | None = None,
) -> StackingClassifier:
    """Build and fit a Stacking Classifier over all traditional models.

    Parameters
    ----------
    models : dict, optional
        Pre-fitted estimators from :func:`train_traditional`.  If *None* the
        models are trained internally (useful if you want the voting classifier
        only).
    """

    if models is None:
        models = train_traditional(X_train, y_train, X_val, y_val)

    calibrated_scv = CalibratedClassifierCV(models['Linear SVC'], cv=None)
    estimators = [
        ('svc',      calibrated_scv),
        ('dt',       models['Decision Tree']),
        ('rf',       models['Random Forest']),
        ('catboost', models['CatBoost']),
        ('lgbm',     models['LightGBM']),
    ]
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), stack_method='predict_proba', n_jobs=n_jobs)
    #training
    print('starting training...')
    stacking_clf.fit(X_train, y_train)
    print('Stacking Classifier trained.')

    #metrics of training
    train_m = compute_metrics(y_train, stacking_clf.predict(X_train), 'Training', 'Stacking Classifier')
    #metrics of validation
    val_m = compute_metrics(y_val, stacking_clf.predict(X_val), 'Validation', 'Stacking Classifier')
    for m in (train_m, val_m):
        print(f"  [{m['dataset']}] Acc={m['accuracy']:.2f}%  "
             f"P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}"
        )

    return stacking_clf

    

    

