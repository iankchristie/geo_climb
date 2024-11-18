import os
import sys
import numpy as np
import rasterio
import torch
from torchgeo.models import RCF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from encoders.dem_dataset import DEMDataset
from utils.file_utils import *
from encoders.preprocess import preprocess_dem


def get_embedding(dem_data, model):
    # Convert the DEM data to a PyTorch tensor
    dem_tensor = dem_data.unsqueeze(0)

    with torch.no_grad():
        embeddings = model(dem_tensor)

    # Flatten the embeddings to a 1D vector
    return embeddings.flatten()


def main(directory, embeddings_dir):
    dem_dataset = DEMDataset()
    model = RCF(
        in_channels=1,
        features=512,
        kernel_size=2,
        seed=42,
        mode="empirical",
        dataset=dem_dataset,
    )
    model.eval()

    for dem_file in os.listdir(directory):
        dem_full_path = os.path.join(directory, dem_file)
        os.makedirs(embeddings_dir, exist_ok=True)
        if dem_file.endswith(".tif"):
            lat, lon = decode_file(dem_file)

            with rasterio.open(dem_full_path) as src:
                dem_data = src.read(1)

            # Normalize the DEM data to sea level
            dem_data = preprocess_dem(dem_data)

            embedding = get_embedding(dem_data, model)
            # print(embedding)
            embeddings_file_path = encode_file(lat, lon, "dem", embeddings_dir, "npy")
            # print(embeddings_file_path)
            np.save(embeddings_file_path, embedding.numpy())


if __name__ == "__main__":
    directory = Config.DATA_DIR_LBL_DEM
    embeddings_dir = "data/labeled/embeddings/dem_v2"
    main(directory, embeddings_dir)

    directory = Config.DATA_DIR_UNLBL_DEM
    embeddings_dir = "data/unlabeled/embeddings/dem_v2"
    main(directory, embeddings_dir)
