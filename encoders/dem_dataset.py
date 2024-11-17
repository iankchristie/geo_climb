import sys
import os
import rasterio
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils.file_utils import *
from torchgeo.datasets import NonGeoDataset


class DEMDataset(NonGeoDataset):
    def __init__(self):
        self.file_paths = []

        for file_name in os.listdir(Config.DATA_DIR_LBL_DEM):
            if file_name.endswith(".tif"):
                file_path = os.path.join(Config.DATA_DIR_LBL_DEM, file_name)
                self.file_paths.append(file_path)

        for file_name in os.listdir(Config.DATA_DIR_UNLBL_DEM):
            if file_name.endswith(".tif"):
                file_path = os.path.join(Config.DATA_DIR_UNLBL_DEM, file_name)
                self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            dem_data = src.read(1)

        # Preprocess the DEM data
        dem_tensor = self.preprocess_dem(dem_data)

        return {"image": dem_tensor}

    def preprocess_dem(self, dem_data):
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


if __name__ == "__main__":
    dataset = DEMDataset()
    h = []
    w = []
    for i in range(dataset.__len__()):
        _, he, we = dataset[i]["image"].shape
        h.append(he)
        w.append(we)
    numbers_h = np.array(h)
    average_h = np.mean(numbers_h)
    print(average_h)

    numbers_w = np.array(w)
    average_w = np.mean(numbers_w)
    print(average_w)
