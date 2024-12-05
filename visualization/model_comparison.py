import os
import sys
from joblib import load
import numpy as np
from sklearn.metrics import cohen_kappa_score
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.geo_climb_data_set import GeoClimbDataset
from model.geo_climb_model import GeoClimbModel


def sort_data_by_lat_lon(data):
    return sorted(data, key=lambda item: (item.latitude, item.longitude))


def prepare_geo_climb_model_predictions(checkpoint_path, dataset):
    sorted_data = sort_data_by_lat_lon(dataset.data)
    embeddings = [item.embeddings for item in sorted_data if not item.labeled]
    tensor_data = torch.tensor(np.array(embeddings), dtype=torch.float32, device="mps")

    model = GeoClimbModel.load_from_checkpoint(
        checkpoint_path, embedding_size=dataset.get_embedding_size()
    )
    model.eval()
    model.to("mps")

    with torch.no_grad():
        predictions = model(tensor_data).cpu().numpy()

    return predictions.flatten(), sorted_data


def prepare_svm_rf_predictions(model_path, dataset):
    sorted_data = sort_data_by_lat_lon(dataset.data)
    embeddings = [item.embeddings for item in sorted_data if not item.labeled]
    embeddings_array = np.array(embeddings)

    model = load(model_path)
    predictions = model.predict_proba(embeddings_array)
    return predictions[:, 1], sorted_data  # Return probabilities for class 1


def compute_kappa_scores(pred_classes_svm, pred_classes_rf, pred_classes_geo_climb):
    kappa_svm_rf = cohen_kappa_score(pred_classes_svm, pred_classes_rf)
    kappa_svm_geo_climb = cohen_kappa_score(pred_classes_svm, pred_classes_geo_climb)
    kappa_rf_geo_climb = cohen_kappa_score(pred_classes_rf, pred_classes_geo_climb)
    return kappa_svm_rf, kappa_svm_geo_climb, kappa_rf_geo_climb


def get_top_predictions(predictions, sorted_data, top_n=10):
    top_indices = np.argsort(-predictions)[:top_n]
    top_coords_probs = [
        (sorted_data[i].latitude, sorted_data[i].longitude, predictions[i])
        for i in top_indices
    ]
    return top_coords_probs


def main():
    checkpoint_path = "geo-climb/3niq79vr/checkpoints/epoch=49-step=27850.ckpt"
    svm_model_path = "geo-climb/SVM_linear.joblib"
    rf_model_path = (
        "geo-climb/RandomForest_{'n_estimators': 10, 'max_depth': None}.joblib"
    )

    geo_climb_dataset = GeoClimbDataset(
        split="test",
        name_encoding="dem_rcf_empirical__lithology_scibert_no_description__sentinel_mosaiks",
    )
    combined_dataset = GeoClimbDataset(split="test", name_encoding="combined")

    geo_climb_predictions, geo_climb_sorted_data = prepare_geo_climb_model_predictions(
        checkpoint_path, geo_climb_dataset
    )
    svm_predictions, svm_sorted_data = prepare_svm_rf_predictions(
        svm_model_path, combined_dataset
    )
    rf_predictions, rf_sorted_data = prepare_svm_rf_predictions(
        rf_model_path, combined_dataset
    )

    predicted_classes_geo_climb = (geo_climb_predictions > 0.5).astype(int)
    predicted_classes_svm = (svm_predictions > 0.5).astype(int)
    predicted_classes_rf = (rf_predictions > 0.5).astype(int)

    kappa_svm_rf, kappa_svm_geo_climb, kappa_rf_geo_climb = compute_kappa_scores(
        predicted_classes_svm, predicted_classes_rf, predicted_classes_geo_climb
    )

    print(f"Cohen's Kappa between SVM and RF: {kappa_svm_rf:.4f}")
    print(f"Cohen's Kappa between SVM and GeoClimbModel: {kappa_svm_geo_climb:.4f}")
    print(f"Cohen's Kappa between RF and GeoClimbModel: {kappa_rf_geo_climb:.4f}")

    avg_predictions = (geo_climb_predictions + svm_predictions + rf_predictions) / 3

    top_geo_climb = get_top_predictions(geo_climb_predictions, geo_climb_sorted_data)
    top_svm = get_top_predictions(svm_predictions, svm_sorted_data)
    top_rf = get_top_predictions(rf_predictions, rf_sorted_data)
    top_avg = get_top_predictions(avg_predictions, geo_climb_sorted_data)

    print("Top 10 GeoClimb Predictions (Lat, Lon, Prob):", top_geo_climb)
    print("Top 10 SVM Predictions (Lat, Lon, Prob):", top_svm)
    print("Top 10 RF Predictions (Lat, Lon, Prob):", top_rf)
    print("Top 10 Averaged Predictions (Lat, Lon, Prob):", top_avg)


if __name__ == "__main__":
    main()
