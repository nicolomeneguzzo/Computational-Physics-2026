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
>>> from stellar_classification.models.network import SimpleNN
>>> from stellar_classification.models.trees import SimpleRandomForest, SimpleExtraTrees
>>> from stellar_classification.experiments.nn_runner import run_experiments, device
>>> from stellar_classification.visualization.nn_plots import plot_training_history, plot_dropout_comparison
>>> from stellar_classification.visualization.model_selection import get_final_candidates, plot_feature_ablation_nn, plot_pca_ablation
>>> from stellar_classification.utils.wrapper import TorchModelWrapper
>>> from stellar_classification.experiments.feature_ablation import run_feature_ablation, get_feature_subset_threshold
>>> from stellar_classification.experiments.pca_ablation import   plot_explained_variance,run_pca_ablation


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
    train_trees_with_tuning,  
    tune_model, 
    evaluate_single_model, 
    make_stacking_classifier,
    train_stacking,
)
from .inference.predictor import (  # noqa: F401
    evaluate_test_set,
    evaluate_neural,
    compute_permutation_importance,
    shap_summary_tree_model, 
    predict_with_confidence 
)
from .visualization import (  # noqa: F401
    plot_class_distribution,
    plot_class_distribution_after_outliers,
    print_outlier_removal_statistics,
    plot_class_distribution_after_smote,
    plot_confusion_matrix,
    plot_permutation_importance,
    plot_feature_ablation,
    plot_prediction_and_error_map, 
    plot_misclassified_feature_distributions, 
    plot_feature_importance, 
    plot_learning_curve,
    plot_misclassified_feature_distributions_separated, 
    plot_misclassified_stacked_hist, 
)
from .utils.metrics import print_metrics  # noqa: F401
from .models.network import SimpleNN     # noqa: F401
from .models.trees import SimpleRandomForest, SimpleExtraTrees
from .experiments.nn_runner import run_experiments, device
from .visualization.nn_plots import plot_training_history, plot_dropout_comparison
from .visualization.model_selection import get_final_candidates, plot_feature_ablation_nn, plot_pca_ablation
from .utils.wrapper import TorchModelWrapper
from .experiments.feature_ablation import run_feature_ablation, get_feature_subset_threshold
from .experiments.pca_ablation import plot_explained_variance, run_pca_ablation


___all__ = [
    # data
    'remove_outliers', 'prepare_splits', 'to_dataloaders',

    # training
    'compute_metrics', 'train_traditional', 'train_voting',
    'train_neural', 'tune_model',
    'train_trees_with_tuning', 'evaluate_single_model',

    # inference
    'evaluate_test_set', 'evaluate_neural',
    'compute_permutation_importance',
    'shap_summary_tree_model', 'predict_with_confidence',

    # visualization
    'plot_class_distribution', 'plot_confusion_matrix',
    'plot_class_distribution_after_outliers',
    'print_outlier_removal_statistics',
    'plot_class_distribution_after_smote'
    'plot_permutation_importance',
    'plot_prediction_and_error_map',
    'plot_misclassified_feature_distributions',
    'plot_feature_importance',
    'plot_misclassified_feature_distributions_separated',
    'plot_misclassified_stacked_hist',
    'plot_feature_ablation', 'plot_learning_curve',
    'plot_pca_ablation',

    # utils
    'print_metrics',

    # models
    'SimpleNN', 'SimpleRandomForest', 'SimpleExtraTrees',

    # experiments
    'run_experiments', 'device',
    'plot_training_history', 'plot_dropout_comparison',
    'get_final_candidates', 'plot_feature_ablation_nn',
    'run_feature_ablation', 'get_feature_subset_threshold',
    'plot_explained_variance', 'run_pca_ablation',
]