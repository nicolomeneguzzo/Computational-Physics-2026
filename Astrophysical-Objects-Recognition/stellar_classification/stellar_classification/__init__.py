"""stellar_classification — top-level package for SDSS stellar-object classification.

Quick-start
-----------
>>> from stellar_classification.data.preprocessing import prepare_splits, to_dataloaders
>>> from stellar_classification.trainer import train_traditional, train_voting, train_neural
>>> from stellar_classification.inference.predictor import (
...     evaluate_test_set, evaluate_neural, compute_permutation_importance, shap_summary_tree_model, predict_with_confidence
... )
>>> from stellar_classification.visualization import (
...     plot_class_distribution, plot_confusion_matrix, plot_permutation_importance,plot_prediction_and_error_map, plot_misclassified_feature_distributions, plot_feature_ablation
... )
>>> from stellar_classification.utils.metrics import print_metrics
"""

from .data.preprocessing import (  # noqa: F401
    remove_outliers,
    prepare_splits,
    to_dataloaders,
)
from .trainer import (  # noqa: F401
    compute_metrics,
    train_traditional,
    tune_model,
    train_voting,
    train_neural,
    evaluate_single_model,
)
from .inference.predictor import (  # noqa: F401
    evaluate_test_set,
    evaluate_neural,
    compute_permutation_importance,
    shap_summary_tree_model,
    predict_with_confidence,
)
from .visualization import (  # noqa: F401
    plot_class_distribution,
    plot_confusion_matrix,
    plot_permutation_importance,
    plot_prediction_and_error_map,
    plot_misclassified_feature_distributions,
    plot_feature_ablation,
)
from .utils.metrics import print_metrics  # noqa: F401
from .models.network import SimpleNN     # noqa: F401

__all__ = [
    # data
    'remove_outliers', 'prepare_splits', 'to_dataloaders',
    # training
    'compute_metrics', 'train_traditional', 'train_voting', 'train_neural',
    # inference
    'evaluate_test_set', 'evaluate_neural', 'compute_permutation_importance', 'shap_summary_tree_model', 'predict_with_confidence',
    # visualization
    'plot_class_distribution', 'plot_confusion_matrix', 'plot_permutation_importance','plot_prediction_and_error_map',
      'plot_misclassified_feature_distributions', 'plot_feature_ablation',
    # utils
    'print_metrics',
    # models
    'SimpleNN',
]
