import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from stellar_classification.data.preprocessing import to_dataloaders
from stellar_classification.experiments.nn_runner import build_model
from stellar_classification.trainer import train_neural
from stellar_classification.inference.predictor import evaluate_neural


def plot_explained_variance(pca):
    """
    Plot cumulative explained variance ratio.
    """

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o"
    )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    plt.title("PCA Explained Variance")

    plt.grid(True)

    plt.show()


def run_pca_ablation(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    best_model_row,
    num_classes,
    device,
    component_list,
    epochs=10,
):
    """
    Train the best model using different PCA dimensions.
    """

    results = []

    best_model_type = best_model_row["model"]
    best_lr = best_model_row["learning_rate"]

    best_dropout = best_model_row.get("dropout", 0.0)

    if pd.isna(best_dropout):
        best_dropout = 0.0

    original_dim = X_train.shape[1]

    for n_components in component_list:

        print(
            f"\nPCA Experiment | "
            f"components={n_components}"
        )

        # ─────────────────────────────
        # PCA
        # ─────────────────────────────

        pca = PCA(n_components=n_components)

        X_train_pca = pca.fit_transform(X_train)
        X_val_pca   = pca.transform(X_val)
        X_test_pca  = pca.transform(X_test)

        explained_variance = (
            pca.explained_variance_ratio_.sum()
        )

        print(
            f"Explained variance: "
            f"{explained_variance * 100:.2f}%"
        )

        # ─────────────────────────────
        # DataLoaders
        # ─────────────────────────────

        train_loader, val_loader, test_loader = to_dataloaders(
            X_train_pca,
            y_train,
            X_val_pca,
            y_val,
            X_test_pca,
            y_test,
            batch_size=64,
        )

        # ─────────────────────────────
        # Build model
        # ─────────────────────────────

        model = build_model(
            best_model_type,
            input_size=n_components,
            num_classes=num_classes,
            dropout=best_dropout,
        )

        # ─────────────────────────────
        # Train
        # ─────────────────────────────

        trained_model, history, final_metrics = train_neural(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=best_lr,
            num_epochs=epochs,
        )

        # ─────────────────────────────
        # Validation
        # ─────────────────────────────

        val_metrics = evaluate_neural(
            val_loader,
            trained_model,
            device,
            model_name=f"PCA_{n_components}",
        )

        # ─────────────────────────────
        # Test
        # ─────────────────────────────

        test_metrics = evaluate_neural(
            test_loader,
            trained_model,
            device,
            model_name=f"PCA_{n_components}",
        )

        results.append({
            "n_components": n_components,
            "explained_variance": explained_variance,
            "val_f1": val_metrics["f1"],
            "test_f1": test_metrics["f1"],
        })

    return pd.DataFrame(results)