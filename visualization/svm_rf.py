import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(y_test, y_pred, probabilities, model_name):
    """
    Visualize true labels vs. predicted labels and their probabilities.
    """
    
    # Sort by predicted probabilities for cleaner visualization
    sorted_indices = np.argsort(probabilities[:, 1])
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    prob_sorted = probabilities[sorted_indices, 1]

    # Plotting y_test vs y_pred with probabilities
    plt.figure(figsize=(12, 6))
    
    # Plot true labels
    plt.scatter(range(len(y_test_sorted)), y_test_sorted, color='blue', alpha=0.6, label="True Labels")
    
    # Plot predicted labels
    plt.scatter(range(len(y_pred_sorted)), y_pred_sorted, color='orange', alpha=0.6, label="Predicted Labels")
    
    # Plot probabilities as line plot
    plt.plot(range(len(prob_sorted)), prob_sorted, color='green', label="Predicted Probabilities", alpha=0.8)
    
    plt.title(f"{model_name}: True vs Predicted Labels with Probabilities")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Class Label / Probability")
    plt.legend()
    plt.show()