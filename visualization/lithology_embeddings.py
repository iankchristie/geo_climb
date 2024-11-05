import os
import sys

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from embeddings import *


if __name__ == "__main__":
    labeled_embeddings, labeled_lat_lon = get_embeddings_and_locations(
        Config.DATA_DIR_LBL_LITH_EMB, "npy"
    )
    unlabeled_embeddings, unlabeled_lat_lon = get_embeddings_and_locations(
        Config.DATA_DIR_UNLBL_LITH_EMB, "npy"
    )

    # plot_labeled_unlabeled_embeddings_pca(
    #     labeled_embeddings, unlabeled_embeddings, UMAP(n_components=2)
    # )

    # plot_embeddings_2d(labeled_embeddings)
    # plot_embeddings_3d(labeled_embeddings)
    # plot_embeddings_map(labeled_embeddings, labeled_lat_lon)

    # plot_embeddings_2d(unlabeled_embeddings)
    # plot_embeddings_3d(unlabeled_embeddings)
    # plot_embeddings_map(unlabeled_embeddings, unlabeled_lat_lon)

    # plot_embeddings_2d(labeled_embeddings, UMAP(n_components=2))
    # plot_embeddings_3d(labeled_embeddings, UMAP(n_components=3))
    # plot_embeddings_map(labeled_embeddings, labeled_lat_lon, UMAP(n_components=3))

    # plot_embeddings_2d(unlabeled_embeddings, UMAP(n_components=2))
    # plot_embeddings_3d(unlabeled_embeddings, UMAP(n_components=3))
    # plot_embeddings_map(unlabeled_embeddings, unlabeled_lat_lon, UMAP(n_components=3))
