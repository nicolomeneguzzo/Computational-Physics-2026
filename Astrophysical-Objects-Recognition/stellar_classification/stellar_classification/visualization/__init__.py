"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance



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
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

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

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

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
    X_test,
    y_test,
    feature_names,
    importance_types=None,
    use_permutation=True,
    n_repeats=10,
    random_state=42,
    normalize=True,
    figsize=(14, 6),
    top_k=None
):
    """
    Universal feature importance plot for:
    - XGBoost
    - LightGBM
    - RandomForest / sklearn models

    Parameters
    ----------
    importance_types : list or None
        If model supports get_booster().get_score, e.g. XGBoost:
        ["weight","gain","total_gain","cover","total_cover"]
        If None, auto-detect sklearn-style importance.
    """

    n_features = len(feature_names)
    results = {}

    # --------------------------
    # 1. TREE-BASED IMPORTANCE
    # --------------------------
    if importance_types is not None and hasattr(model, "get_booster"):
        booster = model.get_booster()

        for t in importance_types:
            imp = booster.get_score(importance_type=t)

            values = np.array([
                imp.get(f"f{i}", 0)
                for i in range(n_features)
            ])

            if normalize and values.sum() > 0:
                values = values / values.sum()

            results[t] = values

    else:
        # sklearn / RF / LGBM style
        if hasattr(model, "feature_importances_"):
            values = np.array(model.feature_importances_)

            if normalize and values.sum() > 0:
                values = values / values.sum()

            results["model_importance"] = values

    # --------------------------
    # 2. PERMUTATION IMPORTANCE
    # --------------------------
    if use_permutation:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        values = perm.importances_mean

        if normalize and values.sum() > 0:
            values = values / values.sum()

        results["permutation"] = values

    # --------------------------
    # 3. DATAFRAME
    # --------------------------
    df = pd.DataFrame(results, index=feature_names)

    # sort by permutation if exists else first column
    sort_col = "permutation" if "permutation" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)

    if top_k is not None:
        df = df.head(top_k)

    # --------------------------
    # 4. PLOT
    # --------------------------
    x = np.arange(len(df.index))
    width = 0.8 / len(df.columns)

    plt.figure(figsize=figsize)

    for i, col in enumerate(df.columns):
        plt.bar(
            x + i * width,
            df[col].values,
            width=width,
            label=col,
            alpha=0.85
        )

    plt.xticks(x + width * (len(df.columns)-1)/2,
               df.index,
               rotation=45,
               ha="right")

    plt.ylabel("Normalized importance" if normalize else "Importance")
    plt.title("Feature Importance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df