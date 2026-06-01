import matplotlib.pyplot as plt


def plot_training_history(history, title: str = "Training History"):
    """
    Plot training curves for a neural network.

    Parameters
    ----------
    history : dict
        Dictionary containing:
        - train_loss
        - train_accuracy
        - val_accuracy
    title : str
        Title of the plots
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    # ─────────────────────────────────────────────────────────────
    # Plot 1: Loss
    # ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")

    plt.title(f"{title} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ─────────────────────────────────────────────────────────────
    # Plot 2: Accuracy
    # ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")

    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_dropout_comparison(history_no_dropout, history_dropout, title: str):
    """
    Compare training curves between no-dropout and dropout models.
    """

    epochs = range(1, len(history_no_dropout["train_loss"]) + 1)

    # ─────────────────────────────────────────────
    # LOSS comparison
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history_no_dropout["train_loss"], label="No Dropout - Train Loss")
    plt.plot(epochs, history_dropout["train_loss"], label="Dropout - Train Loss")

    plt.title(f"{title} - Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

    # ─────────────────────────────────────────────
    # ACCURACY comparison
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history_no_dropout["train_accuracy"], label="No Dropout - Train Acc")
    plt.plot(epochs, history_no_dropout["val_accuracy"], label="No Dropout - Val Acc")

    plt.plot(epochs, history_dropout["train_accuracy"], label="Dropout - Train Acc")
    plt.plot(epochs, history_dropout["val_accuracy"], label="Dropout - Val Acc")

    plt.title(f"{title} - Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.show()