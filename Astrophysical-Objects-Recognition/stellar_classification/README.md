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

#### Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 97.28% | 0.9483 | 0.9616 | 0.9548 |
| Stacking Classifier | 97.51% | 0.9570 | 0.9567 | 0.9569 |
| Neural Network | 96.05% | 0.9277 | 0.9564 | 0.9413 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11508 | 269 | 47 |
| **STAR** | 175 | 1818 | 1 |
| **QUASAR** | 1 | 0 | 4301 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11593 | 208 | 23 |
| **STAR** | 219 | 1775 | 0 |
| **QUASAR** | 2 | 0 | 4300 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11289 | 345 | 190 |
| **STAR** | 162 | 1831 | 1 |
| **QUASAR** | 17 | 0 | 4285 |

---

### Without Redshift

#### Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 88.25% | 0.8452 | 0.8533 | 0.8484 |
| Stacking Classifier | 88.18% | 0.8453 | 0.8479 | 0.8466 |
| Neural Network | 84.01% | 0.7933 | 0.8313 | 0.8091 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11005 | 425 | 401 |
| **STAR** | 258 | 3208 | 320 |
| **QUASAR** | 390 | 546 | 3366 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 11055 | 380 | 396 |
| **STAR** | 315 | 3043 | 428 |
| **QUASAR** | 401 | 435 | 3466 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10109 | 657 | 1065 |
| **STAR** | 211 | 3141 | 434 |
| **QUASAR** | 315 | 503 | 3484 |

---

### Without Redshift, Color Indices: u-g, g-r, r-i, i-z

#### Metrics

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Voting Classifier | 90.78% | 0.8802 | 0.8849 | 0.8818 |
| Stacking Classifier | 90.30% | 0.8745 | 0.8754 | 0.8748 |
| Neural Network | 88.59% | 0.8522 | 0.8596 | 0.8558 |

#### Confusion Matrices

**Voting Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10194 | 364 | 223 |
| **STAR** | 218 | 3114 | 216 |
| **QUASAR** | 270 | 392 | 3266 |

**Stacking Classifier**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10213 | 326 | 242 |
| **STAR** | 253 | 3006 | 289 |
| **QUASAR** | 305 | 356 | 3267 |

**Neural Network**
| | GALAXY | STAR | QUASAR |
|---|---|---|---|
| **GALAXY** | 10009 | 359 | 413 |
| **STAR** | 248 | 2970 | 330 |
| **QUASAR** | 363 | 370 | 3195 |
