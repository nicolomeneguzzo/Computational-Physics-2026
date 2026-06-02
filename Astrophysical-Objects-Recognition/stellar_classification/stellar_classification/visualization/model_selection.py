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


def plot_feature_ablation(df: pd.DataFrame, title: str = "Feature Ablation Study"):
    """
    Plot F1-score vs number of features for ablation study.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["n_features"],
        df["val_f1"],
        marker="o",
        label="Validation F1"
    )

    plt.plot(
        df["n_features"],
        df["test_f1"],
        marker="o",
        label="Test F1"
    )

    plt.xlabel("Number of Features")
    plt.ylabel("F1 Score")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.show()