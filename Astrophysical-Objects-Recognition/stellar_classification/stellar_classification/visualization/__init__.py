"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

#funzione modificata da enrica 
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


#feature ablation ( enrica ) 

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
        m = model.__class__(**model.get_params())
        m.fit(X_train[:, selected_idx], y_train)
        acc = accuracy_score(y_test, m.predict(X_test[:, selected_idx])) * 100
        accuracies.append(acc)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, len(imp_order) + 1), accuracies, marker='o')
    ax.axhline(y=accuracies[-1], color='r', linestyle='--',
               label=f'Full accuracy: {accuracies[-1]:.1f}%')
    ax.set_xlabel('N° features used (most to least important)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title)
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()