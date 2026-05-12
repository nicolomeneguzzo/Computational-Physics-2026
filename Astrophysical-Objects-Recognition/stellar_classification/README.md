# stellar_classification

Stellar object classification pipeline for SDSS data using traditional ML, ensemble voting, and PyTorch neural networks with SHAP interpretability.

## Feature Classes

- **0**: Dwarf Star (main-sequence)
- **1**: Giant Star
- **2**: White Dwarf

## Package Structure

```
stellar_classification/
├── data/preprocessing.py   — Outlier removal, StandardScaler, SMOTE, train/val/test splits, DataLoader creation
├── models/network.py       — SimpleNN (128-hidden linear network)
├── trainer.py              — train_traditional(), train_voting(), train_neural(), compute_metrics()
├── inference/predictor.py  — Test evaluation, permutation importance, SHAP KernelExplainer
├── visualization/          — Confusion matrix, feature importance, class distribution plots
└── utils/metrics.py        — Print-friendly metric formatting
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage from Notebook

```python
import sys
sys.path.insert(0, 'stellar_classification')

from stellar_classification.data.preprocessing import prepare_splits, to_dataloaders
from stellar_classification.models.network import SimpleNN
from stellar_classification.trainer import train_traditional, train_voting, train_neural
from stellar_classification.inference.predictor import evaluate_test_set, evaluate_neural, compute_permutation_importance, compute_shap
from stellar_classification.visualization import plot_confusion_matrix, plot_permutation_importance
from stellar_classification.utils.metrics import print_metrics

# Data pipeline
X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_splits(df)
train_loader, val_loader, test_loader = to_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

# Traditional ML
models = train_traditional(X_train, y_train, X_val, y_val, X_test, y_test)

# Voting ensemble
voting_clf = train_voting(X_train, y_train, X_val, y_val)

# Neural network
nn_model = train_neural(train_loader, val_loader, input_size=X_train.shape[1], num_classes=len(le.classes_))
```

## Training Config

See `configs/training_config.yaml` for hyperparameters.


## work
optimized compute_permutation:importance using batchs methode

---

#Stacking
first problem: linearSVC has no native predict_proba

```calibrated_scv = CalibratedClassifierCV(models['Linear SVC'], cv=None)```

---

##Result

#Con redshift
*Voting Classifier*
Stacking Classifier trained.
  [Training] Acc=97.85%  P=0.98  R=0.98  F1=0.98
  [Validation] Acc=96.88%  P=0.94  R=0.96  F1=0.95
saved file:Astrophysical-Objects-Recognition/notebooks/trained_models/voting_clf_redshift.pkl

*Stacking Classifier*
Stacking Classifier trained.
  [Training] Acc=100.00%  P=1.00  R=1.00  F1=1.00
  [Validation] Acc=97.43%  P=0.95  R=0.96  F1=0.96  
saved file: Astrophysical-Objects-Recognition/notebooks/trained_models/stacking_clf_redshift.pkl

**comparison**

Voting Classifier:
  Accuracy  : 97.28%
  Precision: 0.9483
  Recall   : 0.9616
  F1       : 0.9548
  Confusion Matrix:
[[11508   269    47]
 [  175  1818     1]
 [    1     0  4301]]

Stacking Classifier:
  Accuracy  : 97.51%
  Precision: 0.9570
  Recall   : 0.9567
  F1       : 0.9569
  Confusion Matrix:
[[11593   208    23]
 [  219  1775     0]
 [    2     0  4300]]

Neural Network:
  Accuracy  : 96.05%
  Precision: 0.9277
  Recall   : 0.9564
  F1       : 0.9413
  Confusion Matrix:
[[11289   345   190]
 [  162  1831     1]
 [   17     0  4285]]

#No redshift


Voting Classifier:
  Accuracy  : 88.25%
  Precision: 0.8452
  Recall   : 0.8533
  F1       : 0.8484
  Confusion Matrix:
[[11005   425   401]
 [  258  3208   320]
 [  390   546  3366]]

Stacking Classifier:
  Accuracy  : 88.18%
  Precision: 0.8453
  Recall   : 0.8479
  F1       : 0.8466
  Confusion Matrix:
[[11055   380   396]
 [  315  3043   428]
 [  401   435  3466]]

Neural Network:
  Accuracy  : 84.01%
  Precision: 0.7933
  Recall   : 0.8313
  F1       : 0.8091
  Confusion Matrix:
[[10109   657  1065]
 [  211  3141   434]
 [  315   503  3484]]


#No redshift, index: u-g, g-r, r-i, i-z

Voting Classifier:
  Accuracy  : 90.78%
  Precision: 0.8802
  Recall   : 0.8849
  F1       : 0.8818
  Confusion Matrix:
[[10194   364   223]
 [  218  3114   216]
 [  270   392  3266]]

Stacking Classifier:
  Accuracy  : 90.30%
  Precision: 0.8745
  Recall   : 0.8754
  F1       : 0.8748
  Confusion Matrix:
[[10213   326   242]
 [  253  3006   289]
 [  305   356  3267]]

Neural Network:
  Accuracy  : 88.59%
  Precision: 0.8522
  Recall   : 0.8596
  F1       : 0.8558
  Confusion Matrix:
[[10009   359   413]
 [  248  2970   330]
 [  363   370  3195]]
