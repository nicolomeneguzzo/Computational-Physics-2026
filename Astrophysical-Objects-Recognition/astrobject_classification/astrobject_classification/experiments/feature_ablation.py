import numpy as np
import pandas as pd
import torch

from astrobject_classification.experiments.nn_runner import build_model
from astrobject_classification.trainer import train_neural
from astrobject_classification.inference.predictor import evaluate_neural
from astrobject_classification.data.preprocessing import to_dataloaders
from astrobject_classification.utils.seeding import set_seed

def run_feature_ablation(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_names,
    best_model_row,
    input_size,
    num_classes,
    device,
    importance_scores,
    max_features=None,
    epochs=10,
    seed=42,
):
    """
    Feature ablation study:
    progressively retrains the best model using top-k features.
    """

    set_seed(seed)
 
    # sort features by importance
    ranked_features = importance_scores.sort_values(ascending=False).index.tolist()

    if max_features is None:
        max_features = len(ranked_features)

    results = []

    # get best hyperparameters
    best_model_type = best_model_row["model"]
    best_lr = best_model_row["learning_rate"]

    # Se il modello proviene da run_experiments il valore è NaN
    best_dropout = best_model_row.get("dropout", 0.0)

    if pd.isna(best_dropout):
        best_dropout = 0.0

    print(
        f"\nBest model: "
        f"{best_model_type} | "
        f"lr={best_lr} | "
        f"dropout={best_dropout}"
    )

    # loop over k features
    for k in range(1, max_features + 1):

        if k < len(feature_names):
            selected_features = ranked_features[:k]
        else:
            # ultimo step: usa l'ordine originale delle feature
            selected_features = feature_names

        print(f"\nTraining with top-{k} features: {selected_features}")
        # filter datasets
        feature_to_idx = {f: i for i, f in enumerate(feature_names)}
        idx = [feature_to_idx[f] for f in selected_features]

        X_train_k = X_train[:, idx]
        X_val_k   = X_val[:, idx]
        X_test_k  = X_test[:, idx]
        
        set_seed(seed)
        model = build_model(
            best_model_type,
            input_size=k,
            num_classes=num_classes,
            dropout=best_dropout,
        )

        # train
        train_loader_k, val_loader_k, test_loader_k = to_dataloaders(
            X_train_k, y_train,
            X_val_k, y_val,
            X_test_k, y_test,
            batch_size=64
        )
        
        trained_model, history, final_metrics = train_neural(
            model=model,
            train_loader=train_loader_k,
            val_loader=val_loader_k,
            lr=best_lr,
            num_epochs=epochs,
        )
        # IMPORTANT: evaluation
        val_metrics = evaluate_neural(
            val_loader_k,
            trained_model,
            device,
            model_name=f"top_{k}",
        )

        test_metrics = evaluate_neural(
            test_loader_k,
            trained_model,
            device,
            model_name=f"top_{k}",
        )

        results.append({
            "n_features": k,
            "features": selected_features,
            "val_f1": val_metrics["f1"],
            "test_f1": test_metrics["f1"],
        })

    return pd.DataFrame(results)


def get_feature_subset_threshold(
    ablation_results: pd.DataFrame,
    best_val_f1: float,
    threshold: float = 0.95,
):
    """
    Return the smallest feature subset whose validation F1
    reaches the specified fraction of the full-model F1.

    Parameters
    ----------
    ablation_results : pd.DataFrame
        Output of run_feature_ablation().
    best_val_f1 : float
        Validation F1 of the best full model.
    threshold : float
        Fraction of best_val_f1 to reach (default = 0.95).

    Returns
    -------
    pd.Series
        Row of ablation_results corresponding to the first
        subset satisfying the threshold.
    """

    target = threshold * best_val_f1

    valid_rows = ablation_results[
        ablation_results["val_f1"] >= target
    ]

    if len(valid_rows) == 0:
        return None

    return valid_rows.iloc[0]