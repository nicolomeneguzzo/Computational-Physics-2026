import pandas as pd
import matplotlib.pyplot as plt


def get_final_candidates(results_df, dropout_results, top_k: int = 3, sort_by: str = "val_f1"):
    """
    Combine best models from base training and dropout experiments
    and return a final ranked list of candidates.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from standard training (no dropout)
    dropout_results : pd.DataFrame
        Results from dropout ablation
    top_k : int
        Number of top models to select per group
    sort_by : str
        Metric used for ranking (default: val_f1)

    Returns
    -------
    pd.DataFrame
        Combined and sorted candidate models
    """

    # Top-K base models
    top_base = results_df.sort_values(by=sort_by, ascending=False).head(top_k)

    # Top-K dropout models
    top_dropout = dropout_results.sort_values(by=sort_by, ascending=False).head(top_k)

    # Merge
    combined = pd.concat([top_base, top_dropout], ignore_index=True)

    # Final ranking
    final_sorted = combined.sort_values(by=sort_by, ascending=False)

    return final_sorted




def plot_feature_ablation_nn(
    df: pd.DataFrame,
    best_test_f1: float,
    title: str = "Neural Network: Feature Ablation "):
    """
    Plot F1-score vs number of features for ablation study.
    Also shows baseline (best model) and 95% threshold.
    """

    plt.figure(figsize=(10, 6))

    # Curves
    plt.plot(
        df["n_features"],
        df["test_f1"],
        marker="o",
        label="Test F1"
    )
    
    # Baseline (best model)
    plt.axhline(
        y=best_test_f1,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Full F1 Macro: {best_test_f1:.4f}"
    )
   
    # Labels
    plt.xlabel("Number of Features")
    plt.ylabel("Test F1 Score Macro-score")
    plt.ylim(0.50, 0.90)
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.show()




def plot_pca_ablation(
    df: pd.DataFrame,
    best_val_f1: float,
    title: str = "PCA Ablation Study"
):
    """
    Plot PCA performance vs number of components.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["n_components"],
        df["val_f1"],
        marker="o",
        label="Validation F1",
    )

    plt.axhline(
        y=best_val_f1,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Best Model ({best_val_f1:.4f})"
    )

    plt.xlabel("Number of PCA Components")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()