import os
import sys
import numpy as np
from tqdm import tqdm
import wandb
from joblib import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.geo_climb_data_set import GeoClimbDataset


def negative_predictive_value(preds, targets):
    """
    Compute Negative Predictive Value (NPV).
    Args:
        preds (np.ndarray): Predicted labels (binary: 0 or 1).
        targets (np.ndarray): True labels (binary: 0 or 1).
    Returns:
        float: Negative Predictive Value (NPV).
    """
    tn = np.sum((preds == 0) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    if tn + fn == 0:
        return float("nan")  # Avoid division by zero
    return tn / (tn + fn)


def false_negative_rate_for_negatives(preds, targets):
    """
    Compute the False Negative Rate (FNR) for negative predictions.
    Args:
        preds (np.ndarray): Predicted labels (binary: 0 or 1).
        targets (np.ndarray): True labels (binary: 0 or 1).
    Returns:
        float: False Negative Rate (FNR).
    """
    tn = np.sum((preds == 0) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    if tn + fn == 0:
        return float("nan")  # Avoid division by zero
    return fn / (tn + fn)


def main():
    name = "combined"
    model_name = "RandomForest_{'n_estimators': 100, 'max_depth': 20}"
    model_path = f"geo-climb/{model_name}.joblib"
    geo_climb_dataset = GeoClimbDataset(
        split="test",
        name_encoding=name,
    )

    # Load the saved model
    model = load(model_path)

    preds = []
    targets = []

    for idx in tqdm(range(len(geo_climb_dataset)), desc="Testing"):
        data, label, _, _ = geo_climb_dataset[idx]
        data = data.numpy().reshape(1, -1)  # Convert to numpy and reshape for the model

        prediction = model.predict(data)[0]  # Predict using the loaded model
        preds.append(prediction)
        targets.append(label)

    preds = np.array(preds)
    targets = np.array(targets)

    npv = negative_predictive_value(preds, targets)
    fnr = false_negative_rate_for_negatives(preds, targets)

    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    print(f"False Negative Rate (FNR) for Negative Predictions: {fnr:.4f}")

    # Initialize W&B logging
    wandb.init(project="geo-climb", name=model_name + "_fnr")
    wandb.log({"test_fnr": fnr})
    wandb.finish()


if __name__ == "__main__":
    main()
