import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_interactive_metrics_geo(
    true_positive, false_negative, positive_prediction, negative_prediction
):
    def plot_points(
        ax,
        show_true_positive,
        show_false_negative,
        show_positive_prediction,
        show_negative_prediction,
    ):
        def plot_class(points, color, label):
            # Plot a single point first for the legend. This is so that the legend doesn't contain an item for every point
            if points:
                lat, lon, _ = points[0]
                ax.plot(
                    float(lon),
                    float(lat),
                    marker="o",
                    markersize=6,
                    color=color,
                    transform=ccrs.PlateCarree(),
                    label=label,
                )

            # Plot the rest of the points without a label
            for lat, lon, _ in points[1:]:
                ax.plot(
                    float(lon),
                    float(lat),
                    marker="o",
                    markersize=6,
                    color=color,
                    transform=ccrs.PlateCarree(),
                )

        ax.clear()
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

        # Add geographic features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.STATES, linestyle="-", edgecolor="gray")

        if show_true_positive:
            plot_class(true_positive, "red", "True Positive")

        if show_false_negative:
            plot_class(false_negative, "blue", "False Negative")

        if show_positive_prediction:
            plot_class(positive_prediction, "green", "Positive Prediction")

        if show_negative_prediction:
            plot_class(negative_prediction, "yellow", "Negative Prediction")

        plt.title("Model Metrics")
        ax.legend()
        plt.draw()

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.Mercator()})
    plt.subplots_adjust(left=0.2, right=0.8)
    plot_points(ax, True, True, True, True)

    # Add checkboxes for toggling visibility
    rax = plt.axes([0.05, 0.4, 0.1, 0.15])
    check = CheckButtons(
        rax,
        [
            "True Positive",
            "False Negative",
            "Postive Prediction",
            "Negative_prediction",
        ],
        [True, True, True, True],
    )

    # Event handler for checkboxes
    def update_visibility(label):
        visibility = check.get_status()
        show_true_positive = visibility[0]
        show_false_negative = visibility[1]
        show_positive_prediction = visibility[2]
        show_negative_prediction = visibility[3]

        plot_points(
            ax,
            show_true_positive,
            show_false_negative,
            show_positive_prediction,
            show_negative_prediction,
        )

    check.on_clicked(update_visibility)
    plt.show()


def print_confusion_metrics(
    true_positive, false_negative, positive_prediction, negative_prediction
):
    true_positive_num = len(true_positive)
    false_negative_num = len(false_negative)
    print(f"True Positives: {true_positive_num}")
    print(f"False Negatives: {false_negative_num}")
    sensitivity = true_positive_num / (true_positive_num + false_negative_num)
    print(f"Sensitivity: {sensitivity}")

    positive_prediction_num = len(positive_prediction)
    negative_prediction_num = len(negative_prediction)
    print(f"Positive Predictions: {positive_prediction_num}")
    print(f"Negative Predictions: {negative_prediction_num}")
    prediction_sensitivity = positive_prediction_num / (
        positive_prediction_num + negative_prediction_num
    )
    print(f"Prediction Sensitivity: {prediction_sensitivity}")


def labeled_unlabeled_histogram(labeled_values, unlabeled_values):
    plt.figure(figsize=(10, 6))
    plt.hist(
        labeled_values, bins=20, alpha=0.7, color="blue", label="Labeled (Positive)"
    )
    plt.hist(
        unlabeled_values,
        bins=20,
        alpha=0.7,
        color="orange",
        label="Unlabeled (Unknown)",
    )
    plt.title("Histogram of Model Predictions")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def labeled_unlabeled_roc(labeled_values, unlabeled_values):
    # Create labels for the ROC curve: 1 for labeled, 0 for unlabeled
    labels = np.concatenate(
        [np.ones_like(labeled_values), np.zeros_like(unlabeled_values)]
    )
    predictions = np.concatenate([labeled_values, unlabeled_values])

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.title("ROC Curve: Separation of Labeled vs. Unlabeled Predictions")
    plt.xlabel("Proportion of Unlabeled Data Above Threshold")
    plt.ylabel("Proportion of Labeled Data Above Threshold")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
