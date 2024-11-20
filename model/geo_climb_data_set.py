import os
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import pdb

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import *

LABELED_DIRECTORY = os_safe_file_path("data/labeled/embeddings")
UNLABELED_DIRECTORY = os_safe_file_path("data/unlabeled/embeddings")


@dataclass
class DataItem:
    latitude: str
    longitude: str
    labeled: bool
    embeddings: torch.Tensor


class GeoClimbDataset(Dataset):
    def __init__(self, split, name_encoding: str):
        if split == "training":
            self.file_df = pd.read_csv(os_safe_file_path("data/training_split.csv"))
        elif split == "validation":
            self.file_df = pd.read_csv(os_safe_file_path("data/validation_split.csv"))
        elif split == "test":
            self.file_df = pd.read_csv(os_safe_file_path("data/test_split.csv"))
        else:
            raise Exception("Unknown Split Given")

        self.embedding_directories = name_encoding.split("__")
        self.reverse_index = self.build_reverse_index()
        self.data = self.data_from_csv(self.file_df)

    def build_reverse_index(self):
        result = {}
        for embedding_directory in self.embedding_directories:
            result[embedding_directory] = reverse_index(
                embedding_directory, LABELED_DIRECTORY, UNLABELED_DIRECTORY
            )
        return result

    def load_data(self, latitude, longitude, embedding_directory) -> torch.Tensor:
        try:
            file_path = self.reverse_index[embedding_directory][
                (float(latitude), float(longitude))
            ]
        except KeyError:
            print(
                f"Data not found for latitude {latitude} and longitude {longitude} in {embedding_directory}"
            )
            return torch.tensor([], dtype=torch.float32)

        np_data = np.load(file_path)
        return torch.tensor(np_data, dtype=torch.float32)

    def data_from_csv(self, df: pd.DataFrame) -> list[DataItem]:
        data_items = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Dataset"):
            latitude = str(row["latitude"])
            longitude = str(row["longitude"])
            labeled = bool(row["labeled"])
            embeddings = [
                self.load_data(latitude, longitude, embedding_directory)
                for embedding_directory in self.embedding_directories
            ]
            concatenated_tensor = torch.concat(embeddings)
            data_item = DataItem(
                latitude=latitude,
                longitude=longitude,
                labeled=labeled,
                embeddings=concatenated_tensor,
            )
            data_items.append(data_item)

        return data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: DataItem = self.data[idx]
        label = 1 if row.labeled else self.unlabeled_target()
        return row.embeddings, label, row.latitude, row.longitude

    def unlabeled_target(self):
        return 0

    def get_embedding_size(self) -> int:
        return self.__getitem__(0)[0].shape[0]


if __name__ == "__main__":
    dataset = GeoClimbDataset(
        split="training", name_encoding="dem_v2__lithology_v2__sentinel_mosaiks"
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
