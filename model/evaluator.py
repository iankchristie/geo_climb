import os
import sys
from tqdm import tqdm
import torch
import numpy as np
import wandb

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.model_evaluation import *
from model.geo_climb_model import GeoClimbModel
from model.geo_climb_data_set import GeoClimbDataset


def labeled_unlabeled_analysis(model, test_set, device):
    labeled_values = []
    unlabeled_values = []

    # Disable gradient computation for testing
    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), desc="Testing"):
            data, label, _, _ = test_set[idx]
            data = data.to(device)

            data = data.unsqueeze(0)

            prediction = model(data).item()
            if label == 1:
                labeled_values.append(prediction)
            else:
                unlabeled_values.append(prediction)

    labeled_values = np.array(labeled_values)
    unlabeled_values = np.array(unlabeled_values)
    labeled_unlabeled_histogram(labeled_values, unlabeled_values)
    labeled_unlabeled_roc(labeled_values, unlabeled_values)


def model_performance_analysis(model, test_set, device, threshold=0.9):
    true_positive = []
    false_negative = []
    positive_prediction = []
    negative_prediction = []

    # Disable gradient computation for testing
    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), desc="Testing"):
            data, label, lat, lon = test_set[idx]
            data = data.to(device)

            data = data.unsqueeze(0)

            prediction = model(data).item()
            geo_point = (lat, lon, prediction)
            # USE ROC plotting below to determine the correct threshold.
            # Convert the prediction to a binary label (0 or 1) using a threshold
            predicted_label = 1 if prediction > threshold else 0

            if label == 1:
                if predicted_label == 1:
                    true_positive.append(geo_point)
                else:
                    false_negative.append(geo_point)

            else:
                if predicted_label == 1:
                    positive_prediction.append(geo_point)
                else:
                    negative_prediction.append(geo_point)
    top_positive_predictions = sorted(positive_prediction, key=lambda x: -x[2])
    print(top_positive_predictions)

    print_confusion_metrics(
        true_positive, false_negative, positive_prediction, negative_prediction
    )
    plot_interactive_metrics_geo(
        true_positive, false_negative, positive_prediction, negative_prediction
    )


def evaluate_model(model, test_set):
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA-enabled GPU
        print("Using CUDA for GPU acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Metal Performance Shaders on macOS
        print("Using MPS for GPU acceleration on macOS.")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU as no GPU acceleration is available.")
    model = model.to(device)
    model.eval()

    labeled_unlabeled_analysis(model, test_set, device)
    model_performance_analysis(model, test_set, device=device)


if __name__ == "__main__":
    checkpoint_path = "geo-climb/3niq79vr/checkpoints/epoch=49-step=27850.ckpt"
    name_encoding = (
        "dem_rcf_empirical__lithology_scibert_no_description__sentinel_mosaiks"
    )
    test_set = GeoClimbDataset(split="test", name_encoding=name_encoding)
    model = GeoClimbModel.load_from_checkpoint(
        checkpoint_path,
        embedding_size=test_set.get_embedding_size(),
    )
    evaluate_model(model, test_set)
    # api = wandb.Api()

    # # Specify the project and run ID
    # project = "geo-climb"  # Replace with your project name
    # entity = "iankchristie-cu-boulder"
    # run_id = "1cpwznrp"  # Replace with the specific run ID

    # runs = api.runs(
    #     "iankchristie-cu-boulder/geo-climb"
    # )  # Replace with your entity/project
    # for run in runs:
    #     print(f"Run ID: {run.id}, Name: {run.name}")

    # # Access the run
    # run = api.run(f"{entity}/{project}/{run_id}")

    # # List files in the run
    # for file in run.files():
    #     print(file.name)  # Lists all available files

    # Download a specific checkpoint file
    # checkpoint_file = "model.ckpt"  # Replace with the actual checkpoint file name
