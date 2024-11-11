import os
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


@dataclass
class DataItem:
    latitude: str
    longitude: str
    labeled: bool
    lithology_data: torch.Tensor


class GeoClimbDataset(Dataset):
    def __init__(self, split):
        if split == "training":
            self.file_df = pd.read_csv(Config.DATA_TRAINING_V2)
        elif split == "validation":
            self.file_df = pd.read_csv(Config.DATA_VALIDATION_V2)
        elif split == "test":
            self.file_df = pd.read_csv(Config.DATA_TEST_V2)
        else:
            raise Exception("Unknown Split Given")

        self.data = self.data_from_csv(self.file_df)

    def load_data(self, filepath: str) -> torch.Tensor:
        np_data = np.load(filepath)
        return torch.tensor(np_data, dtype=torch.float32)

    def data_from_csv(self, df: pd.DataFrame) -> list[DataItem]:
        data_items = []

        for _, row in df.iterrows():
            data_item = DataItem(
                latitude=str(row["latitude"]),
                longitude=str(row["longitude"]),
                labeled=bool(row["labeled"]),
                lithology_data=self.load_data(row["lithology_filepath"]),
            )
            data_items.append(data_item)

        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: DataItem = self.data[idx]
        label = 1 if row.labeled else 0
        return row.lithology_data, label, row.latitude, row.longitude
