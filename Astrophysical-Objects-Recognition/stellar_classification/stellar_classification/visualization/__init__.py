"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve 
from matplotlib.lines import Line2D 
from matplotlib.colors import ListedColormap 
from sklearn.metrics import f1_score
from stellar_classification.data.preprocessing import (
    remove_outliers,
    METADATA_COLUMNS,
    Z,
)



def plot_class_distribution(
    y,
    title: str = "Class Distribution"
) -> None:
    """
    Bar chart of class frequencies.

    Parameters
    ----------
    y : array-like
        Target labels (raw strings or encoded integers).
    title : str
        Plot title.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    unique, counts = np.unique(y, return_counts=True)

    bars = ax.bar(
        [str(u) for u in unique],
        counts,
        color=sns.color_palette("tab10", len(unique))
    )

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    # valori sopra le barre
    labels = [f"{c:,}" for c in counts]

    ax.bar_label(
        bars,
        labels=labels,
        padding=3,
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

def plot_class_distribution_after_outliers(
    df,
    target_col="class"
):
    """
    Plot class distribution before and after outlier removal.
    """

    df_before = df.copy()

    df_before.drop(
        columns=METADATA_COLUMNS + Z,
        inplace=True,
        errors="ignore"
    )

    before = (
        df_before[target_col]
        .value_counts()
        .sort_index()
    )

    df_after = remove_outliers(df_before)

    after = (
        df_after[target_col]
        .value_counts()
        .sort_index()
    )

    stats = pd.DataFrame({
        "Before": before,
        "After": after
    }).fillna(0)

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    ax = stats.plot(
        kind="bar",
        figsize=(8, 5),
        width=0.8
    )

    ax.set_title("Class Distribution Before vs After Outlier Removal")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    plt.xticks(rotation=0)
    plt.legend(title="")

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            fontsize=9,
            padding=3
        )

    plt.tight_layout()
    plt.show()


def plot_class_distribution_after_smote(
    df,
    target_col="class",
    test_size=0.2,
    val_ratio=0.25,
    random_state=42,
):
    """
    Plot training-set class distribution
    before and after SMOTE.
    """

    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    df = df.copy()

    # stesso preprocessing di prepare_splits
    df.drop(
        columns=METADATA_COLUMNS + Z,
        inplace=True,
        errors="ignore"
    )

    df = remove_outliers(df)

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # stesso split di prepare_splits
    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_tv
    )

    # distribuzione prima di SMOTE
    before = (
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    # SMOTE
    smote = SMOTE(random_state=1)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    # distribuzione dopo SMOTE
    after = (
        pd.Series(y_train_smote)
        .value_counts()
        .sort_index()
    )

    stats = pd.DataFrame({
        "Before SMOTE": before,
        "After SMOTE": after
    }).fillna(0)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    ax = stats.plot(
        kind="bar",
        figsize=(8, 5),
        width=0.8,
        color=sns.color_palette("bright", 2)
    )

    ax.set_title(
        "Training Set Class Distribution Before vs After SMOTE"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    plt.xticks(rotation=0)
    plt.legend(title="")

    for container in ax.containers:
        labels = [
            f"{int(v):,}"
            for v in container.datavalues
        ]

        ax.bar_label(
            container,
            labels=labels,
            fontsize=9,
            padding=3
        )

    plt.tight_layout()
    plt.show()


def print_outlier_removal_statistics(
    df,
    target_col="class"
):
    df = df.copy()

    # stesso preprocessing di prepare_splits
    df.drop(
        columns=METADATA_COLUMNS + Z,
        inplace=True,
        errors="ignore"
    )

    before = (
        df[target_col]
        .value_counts()
        .sort_index()
    )

    df_clean = remove_outliers(df)

    after = (
        df_clean[target_col]
        .value_counts()
        .sort_index()
    )

    stats = pd.DataFrame({
        "before": before,
        "after": after
    }).fillna(0)

    stats["removed"] = (
        stats["before"] - stats["after"]
    )

    stats["removed_%"] = (
        100 * stats["removed"] / stats["before"]
    )

    print(stats)

    print(
        "\nTotal removed:",
        int(stats["removed"].sum())
    )



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
    ax=None, 
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
    if ax is None:                    
        fig, ax = plt.subplots(figsize=(10, 6))
        standalone = True
    else:
        standalone = False   
    ax.barh(top.index, top.values)
    ax.set_xlabel('Mean Importance')
    ax.set_title(f'{title} (Top {top_n})')
    ax.invert_yaxis()
    if standalone:
        plt.tight_layout()
        plt.show()




def plot_feature_ablation(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    title_prefix="Model",    
    ax=None,
) -> None:
    """Plot F1 Macro-score vs number of features used, from most to least important.
    
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

    # Rank features according to their importance

    imp_order = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(
        ascending=False
    ).index.tolist()

    # Train models using an increasing number of features
    # selected from the most to the least important ones
    f1_scores = []

    for i in range(1, len(imp_order) + 1):

        # Select the first i most important features
        selected_idx = [
            list(feature_names).index(f)
            for f in imp_order[:i]
        ]

        # Create a new model with the same hyperparameters
        m = model.__class__(
            **model.get_params()
        )

        # Train model using only selected features
        m.fit(
            X_train[:, selected_idx],
            y_train
        )

        # Evaluate using macro-averaged F1-score
        f1 = f1_score(
            y_test,
            m.predict(X_test[:, selected_idx]),
            average='macro'
        )

        f1_scores.append(f1)


    # Plot
    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

    ax.plot(
        range(1, len(imp_order) + 1),
        f1_scores,
        marker='o'
    )

    ax.axhline(
        y=f1_scores[-1],
        color='r',
        linestyle='--',
        label=f'Full F1 Macro: {f1_scores[-1]:.3f}'
    )

    ax.set_xlabel('N° features used (most to least important)' )
    ax.set_ylabel('Test F1 Macro-score' )
    ax.set_title(f"{title_prefix}: Feature Ablation")
    ax.legend()


    if standalone:
        plt.tight_layout()
        plt.show()





def plot_prediction_and_error_map(
    X,
    y_true,
    y_pred,
    feature_x,
    feature_y,
    title_prefix="Model",
    s=5,
    alpha=0.5
):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    class_names = {
        0: "Galaxy",
        1: "Quasar",
        2: "Star"
    }

    cmap = ListedColormap([
        "tab:blue",    # galaxy
        "tab:orange",  # quasar
        "tab:green"    # star
    ])

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    # Ground Truth
    ax0.scatter(
        X[feature_x],
        X[feature_y],
        c=y_true,
        cmap=cmap,
        s=s,
        alpha=alpha
    )

    ax0.set_title("Ground Truth")
    ax0.set_xlabel(feature_x)
    ax0.set_ylabel(feature_y)

    # Predictions
    ax1.scatter(
        X[feature_x],
        X[feature_y],
        c=y_pred,
        cmap=cmap,
        s=s,
        alpha=alpha
    )

    ax1.set_title(f"{title_prefix} Predictions")
    ax1.set_xlabel(feature_x)
    ax1.set_ylabel(feature_y)

    # Error map
    errors = np.array(y_pred) != np.array(y_true)

    ax2.scatter(
        X[feature_x],
        X[feature_y],
        c=errors,
        cmap="coolwarm",
        s=s,
        alpha=alpha
    )
    error_legend = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Correct',
        markerfacecolor='blue',
        markersize=8
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='Misclassified',
        markerfacecolor='red',
        markersize=8
    )
    ]

    ax2.legend( handles=error_legend, title="Prediction")
    ax2.set_title(f"{title_prefix} Misclassification Map")
    ax2.set_xlabel(feature_x)
    ax2.set_ylabel(feature_y)

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Galaxy',
            markerfacecolor='tab:blue',
            markersize=8
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Quasar',
            markerfacecolor='tab:orange',
            markersize=8
        ),
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label='Star',
            markerfacecolor='tab:green',
            markersize=8
        )
    ]

    ax0.legend(handles=legend_elements, title="Class")
    ax1.legend(handles=legend_elements, title="Class")

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




def plot_feature_importance(
    model,
    feature_names,
    prefix_name="Model",

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

    # CUSTOM IMPORTANCE FUNCTIONS
    if importance_functions is not None:

        for name, func in importance_functions.items():

            values = np.array(func(model))

            if normalize and values.sum() > 0:
                values = values / values.sum()

            results[name] = values

    # PERMUTATION IMPORTANCE (UNIVERSAL)
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

    #  DATAFRAME
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

    # PLOT
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

    plt.title(f"{prefix_name}: Feature Importance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n Feature Importance Scores:")
    display(df) 
    
    return df





def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    title_prefix="Model",
    cv: int = 5,
    n_points: int = 10,
    ax=None,
) -> None:

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring='f1_macro',
        train_sizes=np.linspace(0.1, 1.0, n_points),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    standalone = ax is None

    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        train_sizes,
        train_mean,
        label='Training F1',
        marker='o'
    )

    ax.plot(
        train_sizes,
        val_mean,
        label='Validation F1',
        marker='o'
    )

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2
    )

    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2
    )

    ax.set_xlabel('Training set size')
    ax.set_ylabel('F1 Macro-score')
    ax.set_title(f"{title_prefix}: Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()




def plot_misclassified_feature_distributions_separated(
    X,
    y_true,
    y_pred,
    features,
    prefix_name="Model",
    class_names=None,
    bins=50,
    figsize=(12,5)
):

    if class_names is None:
        class_names = {
            0: "Galaxy",
            1: "Quasar",
            2: "Star"
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for feat in features:

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        for ax, true_cls in zip(axes, np.unique(y_true)):

            other_classes = [c for c in np.unique(y_true)
                             if c != true_cls]

            colors = ["tab:red", "tab:blue"]

            for pred_cls, col in zip(other_classes, colors):

                mask = (
                    (y_true == true_cls) &
                    (y_pred == pred_cls)
                )

                ax.hist(
                    X.loc[mask, feat],
                    bins=bins,
                    histtype="step",
                    linewidth=2,
                    color=col,
                    label=f"{class_names[true_cls]} → {class_names[pred_cls]}"
                )

            ax.set_title(f"True = {class_names[true_cls]}")
            ax.set_xlabel(feat)
            ax.legend()

        axes[0].set_ylabel("Count")

        fig.suptitle(f"{prefix_name} misclassified objects: {feat}")

        plt.tight_layout()
        plt.show()





def plot_misclassified_stacked_hist(
    X,
    y_true,
    y_pred,
    features,
    prefix_name="Model",
    class_names=None,
    bins=40,
    figsize=(9, 9),
    density=False
):

    if class_names is None:
        class_names = {
            0: "Galaxy",
            1: "Star",
            2: "Quasar"
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 6 tipi di errore
    error_pairs = [
        (0, 1),  # G→S
        (1, 0),  # S→G
        (0, 2),  # G→Q
        (2, 0),  # Q→G
        (1, 2),  # S→Q
        (2, 1),  # Q→S
    ]

    labels = [
        "G→S", "S→G",
        "G→Q", "Q→G",
        "S→Q", "Q→S"
    ]

    colors = [
        "#d62728",  # G→S (red)
        "#ff9896",  # S→G (pink)

        "#1f77b4",  # G→Q (blue)
        "#aec7e8",  # Q→G (light blue)

        "#2ca02c",  # S→Q (green)
        "#98df8a",  # Q→S (light green)
    ]

    n_feat = len(features)

    fig, axes = plt.subplots(
        n_feat,
        1,
        figsize=figsize,
        sharex=True
    )

    if n_feat == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):

        data = []

        for (true_cls, pred_cls) in error_pairs:
            mask = (y_true == true_cls) & (y_pred == pred_cls)
            data.append(X.loc[mask, feat])

        ax.hist(
            data,
            bins=bins,
            stacked=True,
            density=density,
            color=colors,
            alpha=0.9,
            label=labels
        )

        ax.set_xlabel(feat)
        ax.grid(alpha=0.25)

    #axes[-1].set_xlabel("Feature value")
    fig.suptitle(f" {prefix_name} misclassified objects by true class",
    y=0.98  
    )

    fig.legend( labels,
         loc="upper center",
         ncol=6,
         frameon=False,
         bbox_to_anchor=(0.5, 0.94)  
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()