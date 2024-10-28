import json
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare
import numpy as np
from scipy.stats import pearsonr


# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def get_lithology_ids(data):
    return [lith.get("lith_id", "N/A") for lith in data["liths"]]


def get_lithology_data(file_path: str):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        data = data["success"]["data"]["mapData"][0]
        return get_lithology_ids(data)


def analyze(path, title):
    liths = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            liths.append(get_lithology_data(file_path))
    lithology_flat = [litho for sublist in liths for litho in sublist]
    lithology_distribution = Counter(lithology_flat)

    complete_distribution = {}
    for i in range(221):
        if lithology_distribution[i]:
            complete_distribution[i] = lithology_distribution[i]
        else:
            complete_distribution[i] = 0
    print(complete_distribution)

    lithology_distribution_df = pd.DataFrame(
        complete_distribution.items(), columns=["Lithology ID", "Frequency"]
    ).sort_values(by="Lithology ID", ascending=False)

    lithology_distribution_df = pd.DataFrame(
        complete_distribution.items(), columns=["Lithology ID", "Frequency"]
    )
    lithology_distribution_df_sorted = lithology_distribution_df.sort_values(
        by="Lithology ID"
    )

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(
        lithology_distribution_df_sorted["Lithology ID"],
        lithology_distribution_df_sorted["Frequency"],
    )
    plt.xlabel("Lithology ID")
    plt.ylabel("Frequency")
    plt.ylim(0, 5000)  # Set the y-axis maximum value to 5000
    plt.xlim(0, 220)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return complete_distribution


if __name__ == "__main__":
    labeled_dist = analyze(Config.DATA_DIR_LBL_LITH, "Labeled Lithology IDs")
    unlabeled_dist = analyze(Config.DATA_DIR_UNLBL_LITH, "Unlabeled Lithology IDs")

    keys = set(labeled_dist.keys()).union(unlabeled_dist.keys())

    labeled_dist_values = [labeled_dist.get(key, 0) for key in keys]
    unlabeled_dist_values = [unlabeled_dist.get(key, 0) for key in keys]

    # Normalize the two distributions so that their sums are equal
    sum_dist1 = sum(labeled_dist_values)
    sum_dist2 = sum(unlabeled_dist_values)

    # Scale dist2 to match the sum of dist1
    unlabeled_values_normalized = [
        val * (sum_dist1 / sum_dist2) for val in unlabeled_dist_values
    ]

    # Replace zeros in expected values with a small number to avoid division by zero
    unlabeled_values_normalized = np.array(unlabeled_values_normalized)
    unlabeled_values_normalized[unlabeled_values_normalized == 0] = np.finfo(float).eps

    # Calculate Chi-Square distance
    chi_square_value, p_value = chisquare(
        f_obs=labeled_dist_values, f_exp=unlabeled_values_normalized
    )

    print(f"Chi-Square Distance: {chi_square_value}")

    keys = set(labeled_dist.keys()).union(unlabeled_dist.keys())

    labeled_dist_values = [labeled_dist.get(key, 0) for key in keys]
    unlabeled_dist_values = [unlabeled_dist.get(key, 0) for key in keys]

    # Calculate the Pearson correlation coefficient
    correlation_coefficient, p_value = pearsonr(
        labeled_dist_values, unlabeled_dist_values
    )

    print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
