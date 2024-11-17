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


def get_embedding(sen_data, model):
    # Convert the DEM data to a PyTorch tensor
    sen_tensor = (
        torch.tensor(sen_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    )  # Shape: (1, 3, H, W)

    # print(sen_tensor.shape)

    with torch.no_grad():
        embeddings = model(sen_tensor)

    # Flatten the embeddings to a 1D vector
    return embeddings.flatten()


if __name__ == "__main__":
    directory = Config.DATA_DIR_LBL_SEN
    # directory = Config.DATA_DIR_UNLBL_SEN
    embeddings_dir = "data/labeled/embeddings/sentinel_v2"
    os.makedirs(embeddings_dir, exist_ok=True)
    print(directory)

    # Initialize the RCF model
    model = RCF(in_channels=3, features=512, kernel_size=3, seed=42)
    model.eval()

    for sen_file in os.listdir(directory):
        sen_full_path = os.path.join(directory, sen_file)
        if sen_file.endswith(".tif"):
            lat, lon = decode_file(sen_file)

            with rasterio.open(sen_full_path) as src:
                blue = src.read(1)  # Band 1 (B2: Blue)
                green = src.read(2)  # Band 2 (B3: Green)
                red = src.read(3)  # Band 3 (B4: Red)

            # Stack the bands into a 3D array (R, G, B)
            rgb = np.dstack((red, green, blue))

            embedding = get_embedding(rgb, model)
            # print(encode_file(lat, lon, "sen", embeddings_dir, "npy"))
            np.save(
                encode_file(lat, lon, "sen", embeddings_dir, "npy"), embedding.numpy()
            )
