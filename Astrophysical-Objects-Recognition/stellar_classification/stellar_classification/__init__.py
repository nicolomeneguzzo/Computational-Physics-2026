"""stellar_classification — top-level package for SDSS stellar-object classification.

Quick-start
-----------
>>> from stellar_classification.data.preprocessing import prepare_splits, to_dataloaders
>>> from stellar_classification.trainer import train_traditional, train_voting, train_neural, compute_metrics,  tune_model, evaluate_single_model
>>> from stellar_classification.inference.predictor import (
...     evaluate_test_set, evaluate_neural, compute_permutation_importance, shap_summary_tree_model, predict_with_confidence
... )
>>> from stellar_classification.visualization import (
...     plot_class_distribution, plot_confusion_matrix, plot_permutation_importance, 
...     plot_feature_ablation, plot_learning_curve, plot_prediction_and_error_map, plot_misclassified_feature_distributions
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
    train_voting,
    train_neural,
    train_trees_with_tuning,  #aggiunta enrica
    tune_model, #aggiunta sara
    evaluate_single_model, #aggiunta sara
    make_stacking_classifier,
    train_stacking,
)
from .inference.predictor import (  # noqa: F401
    evaluate_test_set,
    evaluate_neural,
    compute_permutation_importance,
    shap_summary_tree_model, #aggiunta sara
    predict_with_confidence #aggiunta sara
)
from .visualization import (  # noqa: F401
    plot_class_distribution,
    plot_confusion_matrix,
    plot_permutation_importance,
    plot_feature_ablation,
    plot_prediction_and_error_map, #sara function
    plot_misclassified_feature_distributions, #sara function
    plot_feature_importance, #sara function
    plot_learning_curve,
    plot_misclassified_feature_distributions_separated, #sara function
    plot_misclassified_stacked_hist, #sara function
)
from .utils.metrics import print_metrics  # noqa: F401
from .models.network import SimpleNN     # noqa: F401
from .models.trees import SimpleRandomForest, SimpleExtraTrees

__all__ = [
    # data
    'remove_outliers', 'prepare_splits', 'to_dataloaders',
    # training
    'compute_metrics', 'train_traditional', 'train_voting', 'train_neural', 'tune_model', 'train_trees_with_tuning', 'evaluate_single_model',
    # inference
    'evaluate_test_set', 'evaluate_neural', 'compute_permutation_importance', 'shap_summary_tree_model', 'predict_with_confidence',
    # visualization
    'plot_class_distribution', 'plot_confusion_matrix', 'plot_permutation_importance', 
    'plot_prediction_and_error_map', 'plot_misclassified_feature_distributions', 'plot_feature_importance', 'plot_misclassified_feature_distributions_separated', 'plot_misclassified_stacked_hist', #-> funzioni di sara 
    'plot_feature_ablation', 'plot_learning_curve', #--> aggiunta enrica 
    # utils
    'print_metrics',
    # models
    'SimpleNN','train_trees_with_tuning', 'evaluate_single_model' #aggiunta enrica e sara
]