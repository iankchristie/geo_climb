import os
import sys
import numpy as np
import rasterio
import torch
from torchgeo.models import RCF
from torchvision.transforms import ToTensor, Resize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from encoders.dem_dataset import DEMDataset
from utils.file_utils import *


def preprocess_dem(dem_data):
    """
    Normalize the DEM data and resize it to (18, 23).
    """
    # Subtract the minimum value to normalize the DEM data
    min_value = np.min(dem_data)
    normalized_data = dem_data - min_value

    # Convert the normalized data to a float32 tensor
    dem_tensor = (
        torch.from_numpy(normalized_data).float().unsqueeze(0)
    )  # Shape: (1, H, W)

    # Resize the tensor to the target size (18, 23)
    resize_transform = Resize((18, 23))
    resized_tensor = resize_transform(dem_tensor.unsqueeze(0)).squeeze(
        0
    )  # Shape: (1, 18, 23)

    return resized_tensor


def get_embedding(dem_data, model):
    # Convert the DEM data to a PyTorch tensor
    dem_tensor = dem_data.unsqueeze(0)

    with torch.no_grad():
        embeddings = model(dem_tensor)

    # Flatten the embeddings to a 1D vector
    return embeddings.flatten()


def main(directory, embeddings_dir):
    dem_dataset = DEMDataset()
    # Initialize the RCF model
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
