import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from geo_climb_data_set import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.svm_rf import *
from tqdm import tqdm

wandb.init(project="geo-climb", name="combined", entity="", config={"random_state": 42})

train_data=GeoClimbDataset("training","combined").data
X_train=[]
y_train=[]

for i in range(len(train_data)):
    if train_data[i].embeddings.shape[0]==0:
        continue
    X_train.append(train_data[i].embeddings.numpy())
    y_train.append(1 if train_data[i].labeled else 0)

X_trainNumpy=np.array(X_train)
y_trainNumpy=np.array(y_train)


test_data=GeoClimbDataset("test","combined").data
X_test=[]
y_test=[]

for i in range(len(test_data)):
    if test_data[i].embeddings.shape[0]==0:
        continue
    X_test.append(test_data[i].embeddings)
    y_test.append(1 if test_data[i].labeled else 0)

X_testNumpy=np.array(X_test)
y_testNumpy=np.array(y_test)

svm = SVC(probability=True, kernel="linear", random_state=42)

print("Training SVM")
# Train the model
svm.fit(X_trainNumpy, y_trainNumpy)

print("Training RF")
# Train a Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_trainNumpy, y_trainNumpy)

# Evaluate both classifiers
models = {"SVM": svm, "Random Forest": rf}

for model_name, model in models.items():
    predictions = []
    for i in tqdm(range(len(X_testNumpy)), desc=f"Predicting using {model_name}"):
        predictions.append(model.predict([X_testNumpy[i]]))
    y_pred = np.array(predictions)
    accuracy = accuracy_score(y_testNumpy, y_pred)
    precision = precision_score(y_testNumpy, y_pred)
    recall = recall_score(y_testNumpy, y_pred)
    f1 = f1_score(y_testNumpy, y_pred)
    roc=roc_auc_score(y_testNumpy,y_pred)

    # Log metrics to W&B
    wandb.log({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc_score":roc
    })

    cm = confusion_matrix(y_testNumpy, y_pred)

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    visualize_predictions(y_testNumpy,y_pred,model.predict_proba(X_testNumpy),model_name)

wandb.finish()