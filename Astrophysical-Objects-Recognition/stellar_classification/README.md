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

## Results

### With Redshift

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 99.07% | 0.99 | 97.15% | 0.95 |
| Stacking Classifier | 100.00% | 1.00 | 97.43% | 0.96 |
| Neural Network | - | - | 95.70% | - |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 97.28% | 0.9489 | 0.9614 | 0.9550 |
| Stacking Classifier | 97.42% | 0.9552 | 0.9554 | 0.9553 |
| Neural Network | 95.92% | 0.9214 | 0.9573 | 0.9381 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11510 | 262 | 52 |
| **STAR** | 176 | 1817 | 1 |
| **QUASAR** | 2 | 0 | 4300 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11583 | 218 | 23 |
| **STAR** | 225 | 1769 | 0 |
| **QUASAR** | 2 | 0 | 4300 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11245 | 416 | 163 |
| **STAR** | 153 | 1839 | 2 |
| **QUASAR** | 5 | 1 | 4296 |

### No Redshift

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 94.75% | 0.95 | 88.56% | 0.85 |
| Stacking Classifier | 100.00% | 1.00 | 88.09% | 0.85 |
| Neural Network | - | - | 86.07% | - |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 88.33% | 0.8459 | 0.8533 | 0.8487 |
| Stacking Classifier | 88.14% | 0.8447 | 0.8456 | 0.8451 |
| Neural Network | 86.22% | 0.8185 | 0.8382 | 0.8272 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11028 | 421 | 382 |
| **STAR** | 259 | 3202 | 325 |
| **QUASAR** | 384 | 554 | 3364 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11089 | 359 | 383 |
| **STAR** | 324 | 3028 | 434 |
| **QUASAR** | 406 | 456 | 3440 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10665 | 590 | 576 |
| **STAR** | 232 | 3160 | 394 |
| **QUASAR** | 418 | 535 | 3349 |

### No Redshift, With SMOTE, Color Indices: u-g, g-r, r-i, i-z

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 96.50% | 0.96 | 90.57% | 0.88 |
| Stacking Classifier | 100.00% | 1.00 | 89.99% | 0.87 |
| Neural Network | - | - | 87.14% | - |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 90.82% | 0.8817 | 0.8844 | 0.8822 |
| Stacking Classifier | 90.11% | 0.8712 | 0.8729 | 0.8719 |
| Neural Network | 87.74% | 0.8406 | 0.8593 | 0.8477 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10215 | 355 | 211 |
| **STAR** | 222 | 3116 | 210 |
| **QUASAR** | 290 | 388 | 3250 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10201 | 334 | 246 |
| **STAR** | 253 | 2981 | 314 |
| **QUASAR** | 300 | 359 | 3269 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 9797 | 552 | 432 |
| **STAR** | 193 | 3135 | 220 |
| **QUASAR** | 332 | 510 | 3086 |

### No SMOT, with redshift

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 99.20% | 0.99 | 97.59% | 0.96 |
| Stacking Classifier | 99.60% | 0.99 | 97.67% | 0.96 |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 97.73% | 0.9702 | 0.9507 | 0.9599 |
| Stacking Classifier | 97.75% | 0.9707 | 0.9505 | 0.9601 |
| Neural Network | 96.58% | 0.9594 | 0.9341 | 0.9454 |
#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11686 | 110 | 28 |
| **STAR** | 271 | 1723 | 0 |
| **QUASAR** | 2 | 0 | 4300 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11693 | 110 | 21 |
| **STAR** | 272 | 1722 | 0 |
| **QUASAR** | 4 | 0 | 4298 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11569 | 97 | 158 |
| **STAR** | 338 | 1654 | 2 |
| **QUASAR** | 24 | 0 | 4278 |


### No Redshift, No SMOTE

#### Base Learners

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Linear SVC | 75.51% | 0.65 | 75.69% | 0.65 |
| Decision Tree | 100.00% | 1.00 | 83.29% | 0.79 |
| Random Forest | 100.00% | 1.00 | 88.76% | 0.85 |
| CatBoost | 92.00% | 0.90 | 88.75% | 0.85 |
| LightGBM | 89.96% | 0.87 | 88.52% | 0.85 |

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 94.38% | 0.93 | 88.80% | 0.85 |
| Stacking Classifier | 97.47% | 0.97 | 89.24% | 0.86 |
| Neural Network | - | - | 86.56% | - |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 88.63% | 0.8605 | 0.8419 | 0.8492 |
| Stacking Classifier | 89.01% | 0.8601 | 0.8502 | 0.8548 |
| Neural Network | 86.36% | 0.8317 | 0.8140 | 0.8212 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11341 | 304 | 186 |
| **STAR** | 351 | 3140 | 295 |
| **QUASAR** | 599 | 529 | 3174 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11279 | 287 | 265 |
| **STAR** | 329 | 3080 | 377 |
| **QUASAR** | 484 | 447 | 3371 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11169 | 335 | 327 |
| **STAR** | 425 | 3013 | 348 |
| **QUASAR** | 721 | 561 | 3020 |



### No Redshift, Color Indices: u-g, g-r, r-i, i-z

#### Training & Validation Metrics

| Model | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| Voting Classifier | 96.50% | 0.96 | 90.57% | 0.88 |
| Stacking Classifier | 100.00% | 1.00 | 89.99% | 0.87 |
| Neural Network | - | - | 87.14% | - |

#### Test Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 90.82% | 0.8817 | 0.8844 | 0.8822 |
| Stacking Classifier | 90.11% | 0.8712 | 0.8729 | 0.8719 |
| Neural Network | 87.74% | 0.8406 | 0.8593 | 0.8477 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10215 | 355 | 211 |
| **STAR** | 222 | 3116 | 210 |
| **QUASAR** | 290 | 388 | 3250 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10201 | 334 | 246 |
| **STAR** | 253 | 2981 | 314 |
| **QUASAR** | 300 | 359 | 3269 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 9797 | 552 | 432 |
| **STAR** | 193 | 3135 | 220 |
| **QUASAR** | 332 | 510 | 3086 |


## GridSearch ##

res:
Best params: {'cv': 10, 'final_estimator__C': 10}

Stacking Classifier trained.
  [Training] Acc=100.00%  P=1.00  R=1.00  F1=1.00
  [Validation] Acc=89.98%  P=0.87  R=0.87  F1=0.87

Stacking Classifier:
  Accuracy  : 90.17%
  Precision: 0.8724
  Recall   : 0.8742
  F1       : 0.8731
  Confusion Matrix:
[[10199   333   249]
 [  250  3001   297]
 [  304   361  3263]]