import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from geo_climb_data_set import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.svm_rf import *
from tqdm import tqdm


def save_model(model, model_name):
    """
    Save the model to a file using joblib.
    """
    filename = f"geo-climb/{model_name}.joblib"
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")


def test_model(model, model_name, config, X_testNumpy, y_testNumpy):
    wandb.init(
        project="geo-climb",
        name=model_name,
        entity="iankchristie-cu-boulder",
        config=config,
    )
    predictions = []
    for i in tqdm(range(len(X_testNumpy)), desc=f"Predicting using {model_name}"):
        predictions.append(model.predict([X_testNumpy[i]]))
    y_pred = np.array(predictions)
    accuracy = accuracy_score(y_testNumpy, y_pred)
    precision = precision_score(y_testNumpy, y_pred)
    recall = recall_score(y_testNumpy, y_pred)
    f1 = f1_score(y_testNumpy, y_pred)
    roc = roc_auc_score(y_testNumpy, y_pred)

    # Log metrics to W&B
    wandb.log(
        {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auroc": roc,
        }
    )
    wandb.finish()

    # cm = confusion_matrix(y_testNumpy, y_pred)
    # cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    # # Visualize the confusion matrix
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm_normalized, display_labels=model.classes_
    # )
    # fig, ax = plt.subplots(figsize=(6, 6))
    # disp.plot(ax=ax, cmap="Blues", values_format=".2f")
    # plt.title(f"Confusion Matrix : {model_name} Classifier")
    # plt.show()

    # visualize_predictions(
    #     y_testNumpy, y_pred, model.predict_proba(X_testNumpy), model_name
    # )


train_data = GeoClimbDataset("training", "combined").data
X_train = []
y_train = []

for i in range(len(train_data)):
    if train_data[i].embeddings.shape[0] == 0:
        continue
    X_train.append(train_data[i].embeddings.numpy())
    y_train.append(1 if train_data[i].labeled else 0)

X_trainNumpy = np.array(X_train)
y_trainNumpy = np.array(y_train)
print(X_trainNumpy, y_trainNumpy)


test_data = GeoClimbDataset("test", "combined").data
X_test = []
y_test = []

for i in range(len(test_data)):
    if test_data[i].embeddings.shape[0] == 0:
        continue
    X_test.append(test_data[i].embeddings)
    y_test.append(1 if test_data[i].labeled else 0)

X_testNumpy = np.array(X_test)
y_testNumpy = np.array(y_test)

# Train and test multiple Random Forest models with different hyperparameters
rf_hyperparams = [
    {"n_estimators": 10, "max_depth": None},
    {"n_estimators": 50, "max_depth": 10},
    {"n_estimators": 100, "max_depth": 20},
]
for params in rf_hyperparams:
    print(f"Training Random Forest with params: {params}")
    rf = RandomForestClassifier(**params, random_state=42)
    rf.fit(X_trainNumpy, y_trainNumpy)
    save_model(rf, f"RandomForest_{params}")
    test_model(rf, f"RandomForest_{params}", params, X_testNumpy, y_testNumpy)


# Train and test multiple SVM models with different kernels
svm_kernels = ["linear", "poly", "rbf", "sigmoid"]
for kernel in svm_kernels:
    print(f"Training SVM with kernel: {kernel}")
    svm = SVC(probability=True, kernel=kernel, random_state=42)
    svm.fit(X_trainNumpy, y_trainNumpy)
    save_model(svm, f"SVM_{kernel}")
    test_model(svm, f"SVM_{kernel}", {"kernel": kernel}, X_testNumpy, y_testNumpy)


# print("Training SVM Linear")
# svm_linear = SVC(probability=True, kernel="linear", random_state=42)
# svm_linear.fit(X_trainNumpy, y_trainNumpy)

# print("Training RF")
# # Train a Random Forest classifier
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_trainNumpy, y_trainNumpy)
