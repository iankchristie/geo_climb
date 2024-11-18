import sys
import os
import rasterio
import numpy as np

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils.file_utils import *
from torchgeo.datasets import NonGeoDataset
from encoders.preprocess import preprocess_sen


class SenDataset(NonGeoDataset):
    def __init__(self):
        self.file_paths = []

        for file_name in os.listdir(Config.DATA_DIR_LBL_SEN):
            if file_name.endswith(".tif"):
                file_path = os.path.join(Config.DATA_DIR_LBL_SEN, file_name)
                self.file_paths.append(file_path)

        for file_name in os.listdir(Config.DATA_DIR_UNLBL_SEN):
            if file_name.endswith(".tif"):
                file_path = os.path.join(Config.DATA_DIR_UNLBL_SEN, file_name)
                self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as src:
            blue = src.read(1)  # Band 1 (B2: Blue)
            green = src.read(2)  # Band 2 (B3: Green)
            red = src.read(3)  # Band 3 (B4: Red)

        # Stack the bands into a 3D array (R, G, B)
        rgb = np.dstack((red, green, blue))

        # Preprocess the DEM data
        sen_tensor = preprocess_sen(rgb)

        return {"image": sen_tensor}


if __name__ == "__main__":
    dataset = SenDataset()
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
