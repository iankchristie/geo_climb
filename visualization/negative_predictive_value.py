import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.geo_climb_data_set import GeoClimbDataset
from model.geo_climb_model import GeoClimbModel


def negative_predictive_value(preds, targets):
    """
    Compute Negative Predictive Value (NPV).
    Args:
        preds (torch.Tensor): Predicted labels (binary: 0 or 1).
        targets (torch.Tensor): True labels (binary: 0 or 1).
    Returns:
        float: Negative Predictive Value (NPV).
    """
    preds, targets = preds.int(), targets.int()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    if tn + fn == 0:
        return float("nan")  # Avoid division by zero
    return tn / (tn + fn)


def false_negative_rate_for_negatives(preds, targets):
    """
    Compute the False Negative Rate (FNR) for negative predictions.
    Args:
        preds (torch.Tensor): Predicted labels (binary: 0 or 1).
        targets (torch.Tensor): True labels (binary: 0 or 1).
    Returns:
        float: False Negative Rate (FNR).
    """
    preds, targets = preds.int(), targets.int()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    if tn + fn == 0:
        return float("nan")  # Avoid division by zero
    return fn / (tn + fn)


def main():
    name = "dem_rcf_empirical__lithology_scibert_no_description"
    checkpoint_path = "geo-climb/67yp12or/checkpoints/epoch=49-step=27850.ckpt"
    geo_climb_dataset = GeoClimbDataset(
        split="test",
        name_encoding=name,
    )

    model = GeoClimbModel.load_from_checkpoint(
        checkpoint_path, embedding_size=geo_climb_dataset.get_embedding_size()
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for idx in tqdm(range(len(geo_climb_dataset)), desc="Testing"):
            data, label, _, _ = geo_climb_dataset[idx]
            data = data.to(device)

            data = data.unsqueeze(0)

            prediction = model(data).item()
            predicted_label = 1 if prediction > 0.5 else 0

            preds.append(predicted_label)
            targets.append(label)

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    npv = negative_predictive_value(preds, targets)
    fnr = false_negative_rate_for_negatives(preds, targets)

    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    print(f"False Negative Rate (FNR) for Negative Predictions: {fnr:.4f}")

    # Initialize W&B logging
    wandb.init(project="geo-climb", name=name + "_fnr")
    wandb.log({"test_fnr": fnr})
    wandb.finish()


if __name__ == "__main__":
    main()
