import os
import sys
import csv
from dataclasses import dataclass, asdict
from typing import List

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils.file_utils import *


@dataclass
class DataItem:
    latitude: str
    longitude: str
    labeled: bool
    lithology_filepath: str


def write_dataclass_list_to_csv(data_items: List[DataItem], csv_filepath: str):
    with open(csv_filepath, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DataItem.__annotations__.keys())
        writer.writeheader()  # Write the header row
        for item in data_items:
            writer.writerow(asdict(item))


if __name__ == "__main__":

    items: list[DataItem] = []

    dir_lith_lab_emb = Config.DATA_DIR_LBL_LITH_EMB_V2
    for file_name in os.listdir(dir_lith_lab_emb):
        if file_name.endswith(".npy"):
            lat, lon = decode_file(file_name)
            file_path = os.path.join(dir_lith_lab_emb, file_name)
            item = DataItem(
                latitude=str(lat),
                longitude=str(lon),
                labeled=True,
                lithology_filepath=file_path,
            )
            items.append(item)

    dir_lith_unlab_emb = Config.DATA_DIR_UNLBL_LITH_EMB_V2
    for file_name in os.listdir(dir_lith_unlab_emb):
        if file_name.endswith(".npy"):
            lat, lon = decode_file(file_name)
            file_path = os.path.join(dir_lith_unlab_emb, file_name)
            item = DataItem(
                latitude=str(lat),
                longitude=str(lon),
                labeled=False,
                lithology_filepath=file_path,
            )
            items.append(item)

    write_dataclass_list_to_csv(items, Config.DATA_AGGREGATION_V2)
