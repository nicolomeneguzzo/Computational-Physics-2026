"""Experiment runner for neural networks hyperparameter tuning."""

import itertools
import pandas as pd
import torch

from stellar_classification.models.network import SimpleNN
from stellar_classification.models.nn_variants import MediumNN, ComplexNN
from stellar_classification.trainer import train_neural
from stellar_classification.inference.predictor import evaluate_neural


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, input_size: int, num_classes: int, dropout: float = 0.0):
    """Create a model based on a string identifier."""

    if model_type == "SimpleNN":
        return SimpleNN(input_size, num_classes)

    elif model_type == "MediumNN":
        return MediumNN(input_size, num_classes, dropout=dropout)

    elif model_type == "ComplexNN":
        return ComplexNN(input_size, num_classes, dropout=dropout)

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
    Run experiments over multiple NN architectures and learning rates.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Default configurations
    # ─────────────────────────────────────────────────────────────────────────

    if model_types is None:
        model_types = ["SimpleNN", "MediumNN", "ComplexNN"]

    if learning_rates is None:
        learning_rates = [0.01, 0.001, 0.0001]

    results = []

    # ─────────────────────────────────────────────────────────────────────────
    # Grid search loop
    # ─────────────────────────────────────────────────────────────────────────

    for model_type, lr in itertools.product(model_types, learning_rates):

        print(f"\nTraining {model_type} | lr={lr}")

        # ─────────────────────────────────────────────────────────────────────
        # 1. Build model
        # ─────────────────────────────────────────────────────────────────────

        model = build_model(model_type, input_size, num_classes)

        # ─────────────────────────────────────────────────────────────────────
        # 2. Train model
        # ─────────────────────────────────────────────────────────────────────

        trained_model, history, final_metrics = train_neural(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            num_epochs=epochs,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 3. Evaluate on TRAIN
        # ─────────────────────────────────────────────────────────────────────

        train_metrics = evaluate_neural(
            train_loader,
            trained_model,
            device,
            model_name=model_type,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 4. Evaluate on VALIDATION
        # ─────────────────────────────────────────────────────────────────────

        val_metrics = evaluate_neural(
            val_loader,
            trained_model,
            device,
            model_name=model_type,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 5. Evaluate on TEST
        # ─────────────────────────────────────────────────────────────────────

        test_metrics = evaluate_neural(
            test_loader,
            trained_model,
            device,
            model_name=model_type,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 6. Save results
        # ─────────────────────────────────────────────────────────────────────

        results.append({

            # Model info
            "model": model_type,
            "learning_rate": lr,

            # Final training info
            "final_loss": final_metrics["final_loss"],
            "final_train_acc": final_metrics["final_train_acc"],
            "final_val_acc": final_metrics["final_val_acc"],

            # Train metrics
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],

            # Validation metrics
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],

            # Test metrics
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],

            # Full training history
            "history": history,
        })

        # ─────────────────────────────────────────────────────────────────────
        # 7. Logging
        # ─────────────────────────────────────────────────────────────────────

        print(
            f"✔ Done: {model_type} | "
            f"lr={lr} | "
            f"loss={final_metrics['final_loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.2f}% | "
            f"val_acc={val_metrics['accuracy']:.2f}% | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Results dataframe
    # ─────────────────────────────────────────────────────────────────────────

    results_df = pd.DataFrame(results)

    # Sort by validation F1
    results_df = results_df.sort_values(by="val_f1", ascending=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Print summaries
    # ─────────────────────────────────────────────────────────────────────────

    print("\nBEST CONFIGURATION:")
    print(results_df.iloc[0])

    print("\nTOP 5 CONFIGURATIONS:")
    print(
        results_df[
            [
                "model",
                "learning_rate",
                "final_loss",
                "train_accuracy",
                "train_f1",
                "val_accuracy",
                "val_f1",
                "test_accuracy",
                "test_f1",
            ]
        ].head(5)
    )

    return results_df

def run_dropout_ablation(
    train_loader,
    val_loader,
    test_loader,
    input_size: int,
    num_classes: int,
    best_models: dict,
    dropout_values=(0.0, 0.3),
    lr: float = 0.001,
    epochs: int = 10,
):
    """
    Run dropout ablation study on best MediumNN and ComplexNN models.

    Parameters
    ----------
    best_models : dict
        Example:
        {
            "MediumNN": True,
            "ComplexNN": True
        }

    dropout_values : tuple
        Dropout probabilities to test.
    """

    results = []

    # Only these architectures
    target_models = ["MediumNN", "ComplexNN"]

    for model_type in target_models:

        # Skip if not provided in best_models
        if best_models is not None and model_type not in best_models:
            continue

        for dropout in dropout_values:

            print(f"\nDropout test → {model_type} | dropout={dropout}")

            # ─────────────────────────────────────────────
            # 1. Build model with dropout
            # ─────────────────────────────────────────────
            model = build_model(
                model_type,
                input_size,
                num_classes,
                dropout=dropout  # IMPORTANT
            )

            # ─────────────────────────────────────────────
            # 2. Train model
            # ─────────────────────────────────────────────
            trained_model, history, final_metrics = train_neural(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                num_epochs=epochs,
            )

            # ─────────────────────────────────────────────
            # 3. Evaluate
            # ─────────────────────────────────────────────
            val_metrics = evaluate_neural(
                val_loader,
                trained_model,
                device,
                model_name=f"{model_type}_dropout_{dropout}"
            )

            test_metrics = evaluate_neural(
                test_loader,
                trained_model,
                device,
                model_name=f"{model_type}_dropout_{dropout}"
            )

            # ─────────────────────────────────────────────
            # 4. Store results
            # ─────────────────────────────────────────────
            results.append({
                "model": model_type,
                "dropout": dropout,

                "final_loss": final_metrics["final_loss"],
                "final_train_acc": final_metrics["final_train_acc"],
                "final_val_acc": final_metrics["final_val_acc"],

                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],

                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],

                "history": history,
            })

            print(
                f"✔ Done: {model_type} | dropout={dropout} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

    # ─────────────────────────────────────────────
    # Summary dataframe
    # ─────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="val_f1", ascending=False)

    print("\nDROPPOUT ABLATION BEST:")
    print(results_df.iloc[0])

    print("\nTOP RESULTS:")
    print(results_df[[
        "model",
        "dropout",
        "final_loss",
        "final_train_acc",
        "final_val_acc",
        "val_accuracy",
        "val_f1",
        "test_accuracy",
        "test_f1",
    ]])

    return results_df