from lightning_module import GeoClimbModel
from data_set import GeoClimbDataset
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

model = GeoClimbModel.load_from_checkpoint(
    "lightning_logs/version_2/checkpoints/epoch=49-step=27850.ckpt"
)
model.eval()

test_set = GeoClimbDataset(split="test")

true_positive = 0
false_negative = 0
positive_prediction = 0
negative_prediction = 0

labeled_values = []
unlabeled_values = []

# Disable gradient computation for testing
with torch.no_grad():
    for idx in tqdm(range(len(test_set)), desc="Testing"):
        data, label = test_set[idx]
        data = data.to(device="mps")

        data = data.unsqueeze(0)

        prediction = model(data).item()
        # USE ROC plotting below to determine the correct threshold.
        # Convert the prediction to a binary label (0 or 1) using a threshold
        threshold = 0.5
        predicted_label = 1 if prediction > threshold else 0

        if label == 1:
            labeled_values.append(prediction)
            if predicted_label == 1:
                true_positive += 1
            else:
                false_negative += 1

        else:
            unlabeled_values.append(prediction)
            if predicted_label == 1:
                positive_prediction += 1
            else:
                negative_prediction += 1

# Print the results
print(f"True Positives: {true_positive}")
print(f"False Negatives: {false_negative}")
sensitivity = true_positive / (true_positive + false_negative)
print(f"Sensitivity: {sensitivity}")
print(f"Positive Predictions: {positive_prediction}")
print(f"Negative Predictions: {negative_prediction}")
prediction_sensitivity = positive_prediction / (
    positive_prediction + negative_prediction
)
print(f"Prediction Sensitivity: {prediction_sensitivity}")


# Convert lists to numpy arrays
labeled_values = np.array(labeled_values)
unlabeled_values = np.array(unlabeled_values)

# Create labels for the ROC curve: 1 for labeled, 0 for unlabeled
labels = np.concatenate([np.ones_like(labeled_values), np.zeros_like(unlabeled_values)])
predictions = np.concatenate([labeled_values, unlabeled_values])

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(labeled_values, bins=20, alpha=0.7, color="blue", label="Labeled (Positive)")
plt.hist(
    unlabeled_values, bins=20, alpha=0.7, color="orange", label="Unlabeled (Unknown)"
)
plt.title("Histogram of Model Predictions")
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
