import pandas as pd

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