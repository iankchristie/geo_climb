import os
import sys
import numpy as np
import rasterio
import torch
from torchgeo.models import RCF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.file_utils import *
from encoders.sen2_dataset import SenDataset
from encoders.preprocess import preprocess_sen


def get_embedding(sen_tensor, model):
    sen_tensor = sen_tensor.unsqueeze(0)
    with torch.no_grad():
        embeddings = model(sen_tensor)

    # Flatten the embeddings to a 1D vector
    return embeddings.flatten()


def main(directory, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)
    sen_dataset = SenDataset()
    model = RCF(
        in_channels=3,
        features=512,
        kernel_size=2,
        seed=42,
        mode="empirical",
        dataset=sen_dataset,
    )
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

            sen_data = preprocess_sen(rgb)

            embedding = get_embedding(sen_data, model)
            embedding_file_path = encode_file(lat, lon, "sen", embeddings_dir, "npy")
            # print(embedding_file_path)
            np.save(embedding_file_path, embedding.numpy())


if __name__ == "__main__":
    directory = Config.DATA_DIR_LBL_SEN
    embeddings_dir = "data/labeled/embeddings/sentinel_rcf_empirical"
    main(directory, embeddings_dir)

    directory = Config.DATA_DIR_UNLBL_SEN
    embeddings_dir = "data/unlabeled/embeddings/sentinel_rcf_empirical"
    main(directory, embeddings_dir)
