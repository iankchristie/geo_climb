import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import sys


# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import *


def normalize_to_rgb(values, min_value=0, max_value=255):
    norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    return (norm_values * (max_value - min_value) + min_value).astype(int)


def plot_embeddings_2d(embeddings, reducer=PCA(n_components=2)):
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot the reduced embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.title("PCA of Embeddings")
    plt.show()


def plot_embeddings_3d(embeddings, reducer=PCA(n_components=3)):
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create an interactive 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        alpha=0.7,
    )

    # Label axes and set title
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.set_title("3D Reduced of Embeddings")

    # Show the plot
    plt.show()


def plot_embeddings_map(embeddings, lat_lon, reducer=PCA(n_components=3)):
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Normalize components to RGB color range
    rgb_colors = np.apply_along_axis(normalize_to_rgb, 0, reduced_embeddings)

    # Plot on a map of the US using Cartopy
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linestyle="-", edgecolor="gray")

    # Plot each point with its RGB color
    for i, (lat, lon) in enumerate(lat_lon):
        color = tuple(
            rgb_colors[i] / 255
        )  # Convert RGB values to 0-1 range for matplotlib
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=6,
            color=color,
            transform=ccrs.PlateCarree(),
        )

    plt.title("Geologic Embeddings Color-coded by Reduced Components")
    plt.show()


def plot_labeled_unlabeled_embeddings_pca(
    labeled_embeddings, unlabeled_embeddings, reducer=PCA(n_components=2)
):
    # Combine labeled and unlabeled embeddings
    all_embeddings = np.vstack((labeled_embeddings, unlabeled_embeddings))
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    # Separate the reduced embeddings back into labeled and unlabeled
    reduced_labeled = reduced_embeddings[: len(labeled_embeddings)]
    reduced_unlabeled = reduced_embeddings[len(labeled_embeddings) :]

    # Plot labeled embeddings in one color
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_labeled[:, 0],
        reduced_labeled[:, 1],
        color="blue",
        alpha=0.7,
        label="Labeled Embeddings",
    )

    # Plot unlabeled embeddings in another color
    plt.scatter(
        reduced_unlabeled[:, 0],
        reduced_unlabeled[:, 1],
        color="red",
        alpha=0.7,
        label="Unlabeled Embeddings",
    )

    # Label axes and add legend
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.title("PCA of Labeled and Unlabeled Embeddings")
    plt.legend()
    plt.show()


def get_embeddings_and_locations(directory, filetype):
    embeddings = []
    lat_lon = []

    for filename in glob.glob(os.path.join(directory, f"*.{filetype}")):
        embeddings.append(np.load(filename))
        lat_lon.append(decode_file(filename))

    embeddings_array = np.array(embeddings)
    lat_lon = np.array(lat_lon)

    return embeddings_array, lat_lon


if __name__ == "__main__":
    # labeled_embeddings, labeled_lat_lon = get_embeddings_and_locations(
    #     Config.DATA_DIR_LBL_LITH_EMB, "npy"
    # )
    # unlabeled_embeddings, unlabeled_lat_lon = get_embeddings_and_locations(
    #     Config.DATA_DIR_UNLBL_LITH_EMB, "npy"
    # )

    labeled_embeddings, labeled_lat_lon = get_embeddings_and_locations(
        "data/labeled/embeddings/sentinel_v2", "npy"
    )

    unlabeled_embeddings, unlabeled_lat_lon = get_embeddings_and_locations(
        "data/unlabeled/embeddings/sentinel_v2", "npy"
    )

    plot_labeled_unlabeled_embeddings_pca(
        labeled_embeddings, unlabeled_embeddings, UMAP(n_components=2)
    )

    plot_embeddings_2d(labeled_embeddings)
    plot_embeddings_3d(labeled_embeddings)
    plot_embeddings_map(labeled_embeddings, labeled_lat_lon)

    plot_embeddings_2d(unlabeled_embeddings)
    plot_embeddings_3d(unlabeled_embeddings)
    plot_embeddings_map(unlabeled_embeddings, unlabeled_lat_lon)

    # plot_embeddings_2d(labeled_embeddings, UMAP(n_components=2))
    # plot_embeddings_3d(labeled_embeddings, UMAP(n_components=3))
    # plot_embeddings_map(labeled_embeddings, labeled_lat_lon, UMAP(n_components=3))

    # plot_embeddings_2d(unlabeled_embeddings, UMAP(n_components=2))
    # plot_embeddings_3d(unlabeled_embeddings, UMAP(n_components=3))
    # plot_embeddings_map(unlabeled_embeddings, unlabeled_lat_lon, UMAP(n_components=3))
