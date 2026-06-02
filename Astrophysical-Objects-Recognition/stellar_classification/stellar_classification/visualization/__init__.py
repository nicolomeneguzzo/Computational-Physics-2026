"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve 


def plot_class_distribution(y, title: str = 'Class Distribution') -> None:
    """Bar chart of class frequencies.

    Parameters
    ----------
    y : array-like
        Target labels (raw strings or encoded integers).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar([str(u) for u in unique], counts, color=sns.color_palette('Set3', len(unique)))
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title: str = 'Confusion Matrix') -> None:
    """Heatmap of a confusion matrix.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix from ``sklearn.metrics.confusion_matrix``.
    class_names : list[str], optional
        Tick labels for axes.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names or 'auto',
        yticklabels=class_names or 'auto',
        ax=ax,
    )
    heatmap.set_title(title)
    heatmap.set_xlabel('Predicted')
    heatmap.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(
    imp: pd.Series,
    top_n: int = 10,
    title: str = 'Permutation Feature Importance',
) -> None:
    """Horizontal bar chart of permutation importances.

    Parameters
    ----------
    imp : pd.Series
        Sorted importances (index = feature names) from
        :func:`~stellar_classification.inference.predictor.compute_permutation_importance`.
    top_n : int
        How many features to display.
    title : str
        Plot title.
    """
    top = imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel('Mean Importance')
    ax.set_title(f'{title} (Top {top_n})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_prediction_and_error_map(
    X,
    y_true,
    y_pred,
    feature_x,
    feature_y,
    title_prefix="Model",
    cmap="viridis",
    error_cmap="coolwarm",
    s=5,
    alpha=0.5
):
    
    # FIX: keep full feature matrix
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

    # Ground truth
    ax0.scatter(X[feature_x], X[feature_y],
                c=y_true, cmap=cmap, s=s, alpha=alpha)
    ax0.set_title("Ground Truth")
    ax0.set_xlabel(feature_x)
    ax0.set_ylabel(feature_y)

    # Predictions
    ax1.scatter(X[feature_x], X[feature_y],
                c=y_pred, cmap=cmap, s=s, alpha=alpha)
    ax1.set_title(f"{title_prefix} Predictions")
    ax1.set_xlabel(feature_x)
    ax1.set_ylabel(feature_y)

    # Errors
    errors = np.array(y_pred) != np.array(y_true)
    ax2.scatter(X[feature_x], X[feature_y],
                c=errors, cmap=error_cmap, s=s, alpha=alpha)
    ax2.set_title("Misclassification map")
    ax2.set_xlabel(feature_x)
    ax2.set_ylabel(feature_y)

    plt.tight_layout()
    plt.show()




def plot_misclassified_feature_distributions(
    X,
    y_true,
    y_pred,
    features,
    bins=50,
    color="crimson",
    alpha=0.7,
    figsize=(12, 8),
    title_prefix="Error"
):
    """
    Plot feature distributions for misclassified samples.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    features : list of str
        Feature names to plot
    bins : int
        Number of histogram bins
    color : str
        Histogram color
    alpha : float
        Transparency
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for subplot titles
    """

   

    # Ensure DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with named features")

    # Misclassified mask
    errors = np.array(y_pred) != np.array(y_true)

    # Filter misclassified samples
    X_err = X[errors]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for i, feat in enumerate(features):
        axes[i].hist(
            X_err[feat],
            bins=bins,
            color=color,
            alpha=alpha
        )
        axes[i].set_title(f"{title_prefix}: distribution of {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Error frequency")
        axes[i].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()    




def plot_feature_ablation(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    title: str = 'Feature Ablation',
    ax=None,
) -> None:
    """Plot accuracy vs number of features used, from most to least important.
    
    Parameters
    ----------
    model : fitted estimator
        Trained sklearn model with feature_importances_ attribute.
    X_train, X_test : np.ndarray
        Training and test arrays.
    y_train, y_test : np.ndarray
        Training and test labels.
    feature_names : list
        Feature names corresponding to columns of X.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        If provided, draws on existing axes. Otherwise creates a new figure.
    """
    from sklearn.metrics import accuracy_score

    imp_order = pd.Series(model.feature_importances_,
                          index=feature_names).sort_values(ascending=False).index.tolist()
    accuracies = []
    for i in range(1, len(imp_order) + 1):
        selected_idx = [list(feature_names).index(f) for f in imp_order[:i]]
        model.fit(X_train[:, selected_idx], y_train)

        preds = model.predict(X_test[:, selected_idx])
        accuracies.append(accuracy_score(y_test, preds) * 100)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, len(imp_order) + 1), accuracies, marker='o')
    ax.axhline(y=accuracies[-1], color='r', linestyle='--',
               label=f'Full accuracy: {accuracies[-1]:.1f}%')
    ax.set_xlabel('N° features used (most to least important)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(50, 100)
    ax.set_title(title)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()





def plot_feature_importance(
    model,
    feature_names,

    # custom importance functions
    importance_functions=None,

    # permutation importance
    use_permutation=False,
    X_test=None,
    y_test=None,
    permutation_scoring=None,
    n_repeats=10,
    random_state=42,

    # plot params
    normalize=True,
    sort_by=None,
    top_k=None,
    figsize=(14, 6),
    rotation=45,
    alpha=0.85,
):
    """
    Fully generalized feature importance plotter.

    Compatible with:
    - XGBoost
    - LightGBM
    - RandomForest
    - Voting
    - Neural Networks
    - sklearn models
    - custom models

    Parameters
    ----------
    model : trained model

    feature_names : list[str]

    importance_functions : dict or None

        Example:
        {
            "gain": lambda m: ...,
            "weight": lambda m: ...,
            "shap": lambda m: ...
        }

    use_permutation : bool
        Automatically adds permutation importance.

    X_test, y_test :
        Required if use_permutation=True
    """

    results = {}

    # =====================================================
    # 1. CUSTOM IMPORTANCE FUNCTIONS
    # =====================================================
    if importance_functions is not None:

        for name, func in importance_functions.items():

            values = np.array(func(model))

            if normalize and values.sum() > 0:
                values = values / values.sum()

            results[name] = values

    # =====================================================
    # 2. PERMUTATION IMPORTANCE (UNIVERSAL)
    # =====================================================
    if use_permutation:

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring=permutation_scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        values = np.array(perm.importances_mean)

        if normalize and values.sum() > 0:
            values = values / values.sum()

        results["permutation"] = values

    # =====================================================
    # 3. DATAFRAME
    # =====================================================
    df = pd.DataFrame(
        results,
        index=feature_names
    )

    # sorting
    if sort_by is None:
        sort_by = df.columns[0]

    df = df.sort_values(
        sort_by,
        ascending=False
    )

    # top k
    if top_k is not None:
        df = df.head(top_k)

    # =====================================================
    # 4. PLOT
    # =====================================================
    x = np.arange(len(df.index))

    width = 0.8 / len(df.columns)

    plt.figure(figsize=figsize)

    for i, col in enumerate(df.columns):

        plt.bar(
            x + i * width,
            df[col].values,
            width=width,
            label=col,
            alpha=alpha
        )

    plt.xticks(
        x + width * (len(df.columns)-1)/2,
        df.index,
        rotation=rotation,
        ha="right"
    )

    plt.ylabel(
        "Normalized importance"
        if normalize
        else "Importance"
    )

    plt.title("Feature Importance Comparison")

    plt.legend()

    plt.tight_layout()

    plt.show()

    return df




def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    title: str = 'Learning Curve',
    cv: int = 5,
    n_points: int = 10,
    ax=None,
) -> None:
    """Plot training and validation accuracy vs training set size.

    Parameters
    ----------
    model : unfitted sklearn estimator
        Fresh model with desired hyperparameters.
    X_train, y_train : np.ndarray
        Training data.
    title : str
        Plot title.
    cv : int
        Number of cross-validation folds.
    n_points : int
        Number of points on the curve.
    ax : matplotlib Axes, optional
        If provided, draws on existing axes.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, n_points),
        n_jobs=3
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_sizes, train_mean, label='Training', marker='o')
    ax.plot(train_sizes, val_mean,   label='Validation', marker='o')
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.2)
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.2)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()