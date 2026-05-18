"""Experiment runner for neural networks hyperparameter tuning."""

import itertools
import pandas as pd
import numpy as np

from stellar_classification.models.network import SimpleNN
from stellar_classification.models.nn_variants import DeepNN, DropoutNN
from stellar_classification.trainer import train_neural
from stellar_classification.inference.predictor import evaluate_test_set


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, input_size: int, num_classes: int):
    """Create a model based on a string identifier."""
    
    if model_type == "SimpleNN":
        return SimpleNN(input_size, num_classes)

    elif model_type == "DeepNN":
        return DeepNN(input_size, num_classes)

    elif model_type == "DropoutNN":
        return DropoutNN(input_size, num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    input_size: int,
    num_classes: int,
    model_types=None,
    learning_rates=None,
    batch_size: int = 64,
    epochs: int = 10,
):
    """
    Run grid search over NN architectures and learning rates.
    """

    if model_types is None:
        model_types = ["SimpleNN", "DeepNN", "DropoutNN"]

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
            train_loader=X_train,
            val_loader=X_val,
            input_size=input_size,
            num_classes=num_classes,
            num_epochs=epochs,
            lr=lr,
        )

        # Nota: train_neural nel tuo progetto crea già il modello internamente
        # quindi qui si assume che venga adattata per accettare un model esterno
        # (oppure si modifica trainer per supportarlo correttamente)

        # 3. evaluate on validation
        val_metrics = evaluate_test_set(
            y_val,
            trained_model.predict(X_val) if hasattr(trained_model, "predict") else None,
            model_name=model_type
        )

        # 4. evaluate on test
        test_metrics = evaluate_test_set(
            y_test,
            trained_model.predict(X_test) if hasattr(trained_model, "predict") else None,
            model_name=model_type
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