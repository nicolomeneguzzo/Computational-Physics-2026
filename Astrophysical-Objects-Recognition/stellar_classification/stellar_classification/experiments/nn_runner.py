"""Experiment runner for neural networks hyperparameter tuning."""

import itertools
import pandas as pd
import numpy as np
import torch

from stellar_classification.models.network import SimpleNN
from stellar_classification.models.nn_variants import MediumNN, ComplexNN
from stellar_classification.trainer import train_neural
from stellar_classification.inference.predictor import evaluate_neural

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, input_size: int, num_classes: int):
    """Create a model based on a string identifier."""
    
    if model_type == "SimpleNN":
        return SimpleNN(input_size, num_classes)

    elif model_type == "MediumNN":
        return MediumNN(input_size, num_classes)

    elif model_type == "ComplexNN":
        return ComplexNN(input_size, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    num_classes: int,
    model_types=None,
    learning_rates=None,
    epochs: int = 10,
):
    """
    Run grid search over NN architectures and learning rates.
    """

    if model_types is None:
        model_types = ["SimpleNN", "MediumNN", "ComplexNN"]

    if learning_rates is None:
        learning_rates = [0.1, 0.01, 0.001]

    results = []

    # ─────────────────────────────────────────────────────────────────────────
    # Grid search loop
    # ─────────────────────────────────────────────────────────────────────────

    for model_type, lr in itertools.product(model_types, learning_rates):

        print(f"\n Training {model_type} | lr={lr}")

        # 1. build model
        model = build_model(model_type, input_size, num_classes)

        # 2. train model
        trained_model = train_neural(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            num_epochs=epochs,
        )


        # 3. evaluate on validation
        val_metrics = evaluate_neural(
            val_loader,
            trained_model,
            device,
            model_name=model_type,
        )

        # 4. evaluate on test
        test_metrics = evaluate_neural(
            test_loader,
            trained_model,
            device,
            model_name=model_type,
        )

        # 5. store results
        results.append({
            "model": model_type,
            "learning_rate": lr,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
        })

        print(f"✔ Done: {model_type} | lr={lr} | F1={val_metrics['f1']:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Results summary
    # ─────────────────────────────────────────────────────────────────────────

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    print("\n BEST CONFIGURATION:")
    print(results_df.iloc[0])

    print("\n TOP 5:")
    print(results_df.head(5))

    return results_df