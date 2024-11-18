import os
import sys
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils.file_utils import *


@dataclass
class DataItem:
    latitude: str
    longitude: str
    labeled: bool
    # lithology_filepath: str | None
    # sentinel2_filepath: str | None
    # dem_filepath: str | None


def write_dataclass_list_to_csv(data_items: List[DataItem], csv_filepath: str):
    with open(csv_filepath, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DataItem.__annotations__.keys())
        writer.writeheader()  # Write the header row
        for item in data_items:
            writer.writerow(asdict(item))


def index_files_by_lat_lon(
    directory: str, labeled: bool
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Index files by (latitude, longitude) from the given directory.
    Returns a dictionary where keys are (lat, lon) tuples, and values are file paths.
    """
    file_index = {}
    print(directory)
    for file_name in os.listdir(directory):
        if file_name.endswith(".npy"):
            lat, lon = decode_file(file_name)
            file_path = os.path.join(directory, file_name)
            key = (str(lat), str(lon))
            if key not in file_index:
                file_index[key] = {
                    "labeled": labeled,
                    "lithology": None,
                    "sentinel2": None,
                    "dem": None,
                }
            # Add the lithology or sentinel2 path based on the directory type
            if "lith" in directory:
                file_index[key]["lithology"] = file_path
            elif "dem" in directory:
                file_index[key]["dem"] = file_path
            else:
                file_index[key]["sentinel2"] = file_path
    return file_index


def split_csv_into_splits(csv_filepath: str):
    df = pd.read_csv(csv_filepath)

    # Shuffle and split the data: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("data/training_split.csv", index=False)
    val_df.to_csv("data/validation_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)
    print("Data split into training.csv, validation.csv, and test.csv")


# Originally this was used to generate splits in a deprecated fashion. The new way is to
# just use the training_split, validation_split, and test_split and dynamically create
# the required files at test time.
if __name__ == "__main__":
    # Index files from all relevant directories
    file_index = {}

    # Index labeled lithology files
    dir_lith_lab_emb = Config.DATA_DIR_LBL_LITH_EMB_V2
    labeled_index = index_files_by_lat_lon(dir_lith_lab_emb, labeled=True)
    file_index.update(labeled_index)

    # Index unlabeled lithology files
    dir_lith_unlab_emb = Config.DATA_DIR_UNLBL_LITH_EMB_V2
    unlabeled_index = index_files_by_lat_lon(dir_lith_unlab_emb, labeled=False)
    file_index.update(unlabeled_index)

    # Index labeled Sentinel-2 files
    dir_sentinel2_lbl_emb = Config.DATA_DIR_LBL_SEN_EMB_V2
    sentinel2_index = index_files_by_lat_lon(dir_sentinel2_lbl_emb, labeled=True)
    for key, value in sentinel2_index.items():
        if key in file_index:
            file_index[key]["sentinel2"] = value["sentinel2"]
        else:
            file_index[key] = value

    # Index unlabled Sentinel-2 files
    dir_sentinel2_unlbl_emb = Config.DATA_DIR_UNLBL_SEN_EMB_V2
    sentinel2_index = index_files_by_lat_lon(dir_sentinel2_unlbl_emb, labeled=False)
    for key, value in sentinel2_index.items():
        if key in file_index:
            file_index[key]["sentinel2"] = value["sentinel2"]
        else:
            file_index[key] = value

    # Index labeled dem files
    dir_dem_lbl_emb = Config.DATA_DIR_LBL_DEM_EMB_V2
    dem_index = index_files_by_lat_lon(dir_dem_lbl_emb, labeled=True)
    for key, value in dem_index.items():
        if key in file_index:
            file_index[key]["dem"] = value["dem"]
        else:
            file_index[key] = value

    # Index unlabled dem files
    dir_dem_unlbl_emb = Config.DATA_DIR_UNLBL_DEM_EMB_V2
    dem_index = index_files_by_lat_lon(dir_dem_unlbl_emb, labeled=False)
    for key, value in dem_index.items():
        if key in file_index:
            file_index[key]["dem"] = value["dem"]
        else:
            file_index[key] = value

    # Convert indexed data into a list of DataItem objects
    items: list[DataItem] = []
    for (lat, lon), data in file_index.items():
        item = DataItem(
            latitude=lat,
            longitude=lon,
            labeled=data["labeled"],
            # lithology_filepath=data.get("lithology"),
            # sentinel2_filepath=data.get("sentinel2"),
            # dem_filepath=data.get("dem"),
        )
        items.append(item)

    # for d in items:
    #     if not d.lithology_filepath or not d.sentinel2_filepath:
    #         print(d)

    # Write the aggregated data to CSV
    write_dataclass_list_to_csv(items, "data/all_lat_lons.csv")
    split_csv_into_splits("data/all_lat_lons.csv")
