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
    sentinel_data: torch.Tensor
    dem_data: torch.Tensor


class GeoClimbDataset(Dataset):
    def __init__(self, split, data_types=["lithology", "sentinel", "dem"]):
        if split == "training":
            self.file_df = pd.read_csv(Config.DATA_TRAINING_V5)
        elif split == "validation":
            self.file_df = pd.read_csv(Config.DATA_VALIDATION_V5)
        elif split == "test":
            self.file_df = pd.read_csv(Config.DATA_TEST_V5)
        else:
            raise Exception("Unknown Split Given")

        self.data = self.data_from_csv(self.file_df)
        self.data_types = data_types

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
                sentinel_data=self.load_data(row["sentinel2_filepath"]),
                dem_data=self.load_data(row["dem_filepath"]),
            )
            data_items.append(data_item)

        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: DataItem = self.data[idx]
        label = 1 if row.labeled else 0
        return self.get_concatenated_tensor(row), label, row.latitude, row.longitude

    def get_concatenated_tensor(self, data_item: DataItem) -> torch.Tensor:
        """
        Given a DataItem and a list of data_types, extract the corresponding tensors and concatenate them.
        """
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
    dem_dataset = GeoClimbDataset(split="training", data_types=["dem"])
    print(dem_dataset.get_embedding_size())

    # lith_dataset = GeoClimbDataset(split="training", data_types=["lithology"])
    # print(lith_dataset.get_embedding_size())

    # sen_dataset = GeoClimbDataset(split="training", data_types=["sentinel"])
    # print(sen_dataset.get_embedding_size())

    # both_dataset = GeoClimbDataset(
    #     split="training", data_types=["lithology", "sentinel"]
    # )
    # print(both_dataset.get_embedding_size())
