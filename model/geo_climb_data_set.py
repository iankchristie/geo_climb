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
from utils.file_utils import *

SENTINEL_DIRECTORY = "sentinel_v2"
DEM_DIRECTORY = "dem_v2"
LITHOLOGY_DIRECTORY = "lithology_v2"

LABELED_DIRECTORY = os_safe_file_path("data/labeled/embeddings")
UNLABELED_DIRECTORY = os_safe_file_path("data/unlabeled/embeddings")


@dataclass
class DataItem:
    latitude: str
    longitude: str
    labeled: bool
    lithology_data: torch.Tensor | None
    sentinel_data: torch.Tensor | None
    dem_data: torch.Tensor | None


class GeoClimbDataset(Dataset):
    def __init__(self, split, data_types=["lithology", "sentinel", "dem"]):
        if split == "training":
            self.file_df = pd.read_csv(os_safe_file_path("data/training_split.csv"))
        elif split == "validation":
            self.file_df = pd.read_csv(os_safe_file_path("data/validation_split.csv"))
        elif split == "test":
            self.file_df = pd.read_csv(os_safe_file_path("data/test_split.csv"))
        else:
            raise Exception("Unknown Split Given")

        self.data_types = data_types
        self.data = self.data_from_csv(self.file_df)

    def load_data(self, row, type) -> torch.Tensor:
        latitude = str(row["latitude"])
        longitude = str(row["longitude"])
        labeled = bool(row["labeled"])

        if labeled:
            directory = LABELED_DIRECTORY
        else:
            directory = UNLABELED_DIRECTORY

        if type == "sen":
            file_path = os.path.join(directory, SENTINEL_DIRECTORY)
        elif type == "dem":
            file_path = os.path.join(directory, DEM_DIRECTORY)
        else:
            file_path = os.path.join(directory, LITHOLOGY_DIRECTORY)

        full_file_path = encode_file(latitude, longitude, type, file_path, "npy")

        np_data = np.load(full_file_path)
        return torch.tensor(np_data, dtype=torch.float32)

    def data_from_csv(self, df: pd.DataFrame) -> list[DataItem]:
        data_items = []

        for _, row in df.iterrows():
            data_item = DataItem(
                latitude=str(row["latitude"]),
                longitude=str(row["longitude"]),
                labeled=bool(row["labeled"]),
                lithology_data=(
                    self.load_data(row, "lit")
                    if "lithology" in self.data_types
                    else None
                ),
                sentinel_data=(
                    self.load_data(row, "sen")
                    if "sentinel" in self.data_types
                    else None
                ),
                dem_data=(
                    self.load_data(row, "dem") if "dem" in self.data_types else None
                ),
            )
            data_items.append(data_item)

        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: DataItem = self.data[idx]
        label = 1 if row.labeled else self.unlabeled_target()
        return self.get_concatenated_tensor(row), label, row.latitude, row.longitude

    def unlabeled_target(self):
        return 0

    def get_concatenated_tensor(self, data_item: DataItem) -> torch.Tensor:
        data_mapping = {
            "lithology": data_item.lithology_data,
            "sentinel": data_item.sentinel_data,
            "dem": data_item.dem_data,
        }

        # Extract tensors based on the data_types list
        tensors_to_concat = [
            data_mapping[data_type]
            for data_type in self.data_types
            if data_type in data_mapping
        ]

        # Concatenate the selected tensors along the feature dimension (usually dim=1 for 2D tensors)
        if tensors_to_concat:
            concatenated_tensor = torch.cat(tensors_to_concat, dim=0)
        else:
            raise ValueError("No valid data types provided for concatenation.")

        return concatenated_tensor

    def get_embedding_size(self) -> int:
        return self.__getitem__(0)[0].shape[0]


if __name__ == "__main__":
    dataset = GeoClimbDataset(
        split="training", data_types=["dem", "sentinel", "lithology"]
    )
    print(dataset.get_embedding_size())

    # lith_dataset = GeoClimbDataset(split="training", data_types=["lithology"])
    # print(lith_dataset.get_embedding_size())

    # sen_dataset = GeoClimbDataset(split="training", data_types=["sentinel"])
    # print(sen_dataset.get_embedding_size())

    # both_dataset = GeoClimbDataset(
    #     split="training", data_types=["lithology", "sentinel"]
    # )
    # print(both_dataset.get_embedding_size())
