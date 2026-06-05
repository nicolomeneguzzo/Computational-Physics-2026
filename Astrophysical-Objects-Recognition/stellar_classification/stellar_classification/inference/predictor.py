"""Inference helpers: test evaluation and feature importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import shap
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_test_set(
    y_test,
    y_pred,
    model_name: str = 'Model',
) -> dict:
    """Compute test metrics and return as a dict."""
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def evaluate_neural(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    model_name: str = 'Neural Network',
) -> dict:
    """Run inference on test loader and return metrics."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            _, pred = model(Xb).max(1)
            preds.extend(pred.cpu().numpy())
            trues.extend(yb.numpy())
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    return {
        'model': model_name,
        'accuracy': accuracy_score(trues, preds) * 100,
        'precision': precision_score(trues, preds, average='macro'),
        'recall': recall_score(trues, preds, average='macro'),
        'f1': f1_score(trues, preds, average='macro'),
        'confusion_matrix': confusion_matrix(trues, preds),
    }


def compute_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance and return sorted DataFrame."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    return imp


def compute_shap(
    model,
    X_train_background,
    X_test_subset,
    feature_names,
    background_size: int = 100,
) -> shap.KernelExplainer:
    """Return a SHAP KernelExplainer fitted on the model."""
    def predict_fn(X):
        return model.predict(X)

    explainer = shap.KernelExplainer(predict_fn, X_train_background[:background_size])
    shap_values = explainer.shap_values(X_test_subset)
    return explainer, shap_values


def shap_summary_tree_model(model, X, feature_names, class_names=None, show=True):
    """
    Generate SHAP summary plots for tree-based multi-class models.

    Parameters
    ----------
    model : fitted tree-based model (e.g., XGBoost, LightGBM)
    X : array-like or pd.DataFrame
        Input features (test set)
    feature_names : list
        Names of features
    class_names : list, optional
        Class labels (from LabelEncoder)
    show : bool, default=True
        Whether to display plots

    Returns
    -------
    shap_values : computed SHAP values
    """

    # Ensure DataFrame format
    X_df = pd.DataFrame(X, columns=feature_names)

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_df)

    # Class names fallback
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(shap_values[0]))]

    # Plot per class
    for i, cls in enumerate(class_names):
        print(f"\nSHAP summary for class: {cls}")
        shap.summary_plot(shap_values[:, :, i], X_df, show=show)

    return shap_values



def predict_with_confidence(
    model,
    X_test,
    feature_x,
    feature_y,
    title_prefix="Model",
    X_plot=None,
    threshold=0.9,
    uncertain_label=-1,
    plot=True,
    cmap="viridis",
):
    """
    Predict classes using model.predict_proba() and optionally plot confidence.

    Parameters
    ----------
    model : estimator
        Trained classifier with predict_proba() method.
    X_test : array-like or DataFrame
        Input data used for prediction.
    X_plot : DataFrame, optional
        Dataset used for plotting. If None, X_test is used.
    feature_x : str, default="g_r"
        Column name for x-axis in scatter plot.
    feature_y : str, default="r_i"
        Column name for y-axis in scatter plot.
    threshold : float, default=0.9
        Confidence threshold below which predictions are marked uncertain.
    uncertain_label : int, default=-1
        Label assigned to uncertain predictions.
    plot : bool, default=True
        Whether to generate the confidence scatter plot.
    cmap : str, default="viridis"
        Colormap for scatter plot.

    Returns
    -------
    dict
        Dictionary containing:
        - probabilities
        - confidence
        - predictions
        - final_predictions
    """

    # Predict probabilities
    probs = model.predict_proba(X_test)

    # Confidence and predicted class
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    # Apply threshold
    final_predictions = np.where(
        confidence > threshold,
        predictions,
        uncertain_label,
    )

    # Plot
    if plot:
        if X_plot is None:
            X_plot = X_test

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_plot[feature_x],
            X_plot[feature_y],
            c=confidence,
            cmap=cmap,
        )

        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.colorbar(scatter, label="Confidence")
        plt.title(f"{title_prefix} Prediction Confidence")
        plt.show()

    return {
        "probabilities": probs,
        "confidence": confidence,
        "predictions": predictions,
        "final_predictions": final_predictions,
    }


def predict_with_qso_threshold(model, X, threshold=0.6):

    probs = model.predict_proba(X)

    preds = []

    for p in probs:

        p_galaxy = p[0]
        p_quasar = p[1]
        p_star   = p[2]

        if p_quasar > threshold:

            preds.append(1)

        else:

            if p_galaxy > p_star:
                preds.append(0)

            else:
                preds.append(2)

    return np.array(preds)