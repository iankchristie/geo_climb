import json
import os
import sys
import rasterio
import numpy as np

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloaders.file_utils import *


def compare_tif_files(dir1: str, dir2: str):
    """
    This was used to ensure that there wasn't a bug in the parallel downloading by checkign that the
    synchronous and parallel downloading produced the same results.
    """
    mismatched_files = []

    # Get the list of .tif files in the first directory
    files_dir1 = {f for f in os.listdir(dir1) if f.endswith(".tif")}
    files_dir2 = {f for f in os.listdir(dir2) if f.endswith(".tif")}

    # Find the common files between the two directories
    common_files = files_dir1.intersection(files_dir2)

    print(f"comparing {len(common_files)} files")

    # Compare each common .tif file
    for file_name in common_files:
        file1_path = os.path.join(dir1, file_name)
        file2_path = os.path.join(dir2, file_name)

        # Open the files using rasterio
        with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
            # Check that the dimensions (width, height) are the same
            if src1.width != src2.width or src1.height != src2.height:
                mismatched_files.append(file_name)
                continue

            # Check that the number of bands are the same
            if src1.count != src2.count:
                mismatched_files.append(file_name)
                continue

            # Compare the metadata
            if src1.meta != src2.meta:
                mismatched_files.append(file_name)
                continue

            # Compare the pixel data
            for i in range(1, src1.count + 1):  # Loop through each band
                band1 = src1.read(i)
                band2 = src2.read(i)

                if not np.array_equal(band1, band2):
                    mismatched_files.append(file_name)
                    break
    return mismatched_files


def get_file_name_without_extension(file_name):
    """
    Return the file name without its extension.
    """
    base_name = os.path.splitext(file_name)[0]
    return base_name[4:]  # Remove the first 4 characters that specify the data type.


def compare_directories(dir1, dir2, dir3):
    """
    Compare the file names in three directories, print out the differences,
    and also print files that are uniquely missing in each directory.

    Parameters:
    - dir1, dir2, dir3: Paths to the directories to compare.
    """
    # Get the set of file names without extensions in each directory
    files_in_dir1 = set(get_file_name_without_extension(f) for f in os.listdir(dir1))
    files_in_dir2 = set(get_file_name_without_extension(f) for f in os.listdir(dir2))
    files_in_dir3 = set(get_file_name_without_extension(f) for f in os.listdir(dir3))

    # Find files that are in dir1 but not in dir2 or dir3
    only_in_dir1 = files_in_dir1 - files_in_dir2 - files_in_dir3
    if only_in_dir1:
        print(f"Files only in {dir1}:")
        print("\n".join(only_in_dir1))
    else:
        print(f"No unique files in {dir1}")

    # Find files that are in dir2 but not in dir1 or dir3
    only_in_dir2 = files_in_dir2 - files_in_dir1 - files_in_dir3
    if only_in_dir2:
        print(f"Files only in {dir2}:")
        print("\n".join(only_in_dir2))
    else:
        print(f"No unique files in {dir2}")

    # Find files that are in dir3 but not in dir1 or dir2
    only_in_dir3 = files_in_dir3 - files_in_dir1 - files_in_dir2
    if only_in_dir3:
        print(f"Files only in {dir3}:")
        print("\n".join(only_in_dir3))
    else:
        print(f"No unique files in {dir3}")

    # Find files missing in dir1 but present in dir2 and dir3
    missing_in_dir1 = (files_in_dir2 & files_in_dir3) - files_in_dir1
    if missing_in_dir1:
        print(f"Files missing in {dir1} but present in {dir2} and {dir3}:")
        print("\n".join(missing_in_dir1))
    else:
        print(f"No files uniquely missing in {dir1}")

    # Find files missing in dir2 but present in dir1 and dir3
    missing_in_dir2 = (files_in_dir1 & files_in_dir3) - files_in_dir2
    if missing_in_dir2:
        print(f"Files missing in {dir2} but present in {dir1} and {dir3}:")
        print("\n".join(missing_in_dir2))
    else:
        print(f"No files uniquely missing in {dir2}")

    # Find files missing in dir3 but present in dir1 and dir2
    missing_in_dir3 = (files_in_dir1 & files_in_dir2) - files_in_dir3
    if missing_in_dir3:
        print(f"Files missing in {dir3} but present in {dir1} and {dir2}:")
        print("\n".join(missing_in_dir3))
    else:
        print(f"No files uniquely missing in {dir3}")


# Warning: dem_37.59522_-122.52383.tif contains no valid data
# Warning: dem_42.26058_-70.88709.tif contains no valid data
def check_dem_files(directory):
    """
    Check each .tif file in the given directory to ensure it has data and only one band.

    Parameters:
    - directory: Path to the directory containing .tif files.

    Prints the result of the checks for each .tif file.
    """
    invalid = []
    # List all .tif files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            file_path = os.path.join(directory, filename)
            try:
                # Open the .tif file with rasterio
                with rasterio.open(file_path) as src:
                    # Check the number of bands
                    band_count = src.count
                    if band_count != 1:
                        print(
                            f"Warning: {filename} contains {band_count} bands (should be 1)"
                        )
                        invalid.append(filename)
                        continue

                    # Check if there is data in the file
                    data = src.read(1)  # Read the first (and only) band
                    if not data.any():
                        print(f"Warning: {filename} contains no valid data")
                        invalid.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                invalid.append(filename)
    return invalid


# Warning: Bands of sen_43.99828_-71.19971.tif contains no valid data
# Warning: Bands of sen_35.86384_-93.04812.tif contains no valid data
# Warning: Bands of sen_34.21976_-116.9928.tif contains no valid data
# Warning: Bands of sen_34.27148_-116.98311.tif contains no valid data
# Warning: Bands of sen_34.21994_-116.99222.tif contains no valid data
# Warning: Bands of sen_44.0447_-71.3618.tif contains no valid data
# Warning: Bands of sen_34.27822_-116.9698.tif contains no valid data
# Warning: Bands of sen_44.00694_-71.23067.tif contains no valid data
# Warning: Bands of sen_34.21992_-116.99268.tif contains no valid data
# Warning: Bands of sen_44.00669_-71.23158.tif contains no valid data
# Warning: Bands of sen_44.03984_-71.39582.tif contains no valid data
# Warning: Bands of sen_43.99824_-71.20047.tif contains no valid data
# Warning: Bands of sen_43.99821_-71.20059.tif contains no valid data
# Warning: Bands of sen_44.01938_-71.25702.tif contains no valid data
# Warning: Bands of sen_43.99844_-71.19998.tif contains no valid data
# Warning: Bands of sen_44.00839_-71.23031.tif contains no valid data
# Warning: Bands of sen_44.043_-71.3591.tif contains no valid data
# Warning: Bands of sen_43.9986_-71.19948.tif contains no valid data
# Warning: Bands of sen_44.01767_-71.26066.tif contains no valid data
# Warning: Bands of sen_44.00903_-71.2253.tif contains no valid data
# Warning: Bands of sen_44.02081_-71.25632.tif contains no valid data
# Warning: Bands of sen_35.84203_-93.05528.tif contains no valid data
# Warning: Bands of sen_34.2276_-116.9883.tif contains no valid data
# Warning: Bands of sen_35.8416_-93.0534.tif contains no valid data
# Warning: Bands of sen_44.02013_-71.25665.tif contains no valid data
# Warning: Bands of sen_34.21983_-116.99266.tif contains no valid data
# Warning: Bands of sen_44.04497_-71.36221.tif contains no valid data
# Warning: Bands of sen_34.21967_-116.99275.tif contains no valid data
def check_sen_files(directory):
    """
    Check each .tif file in the given directory to ensure it has data and exactly 3 bands.

    Parameters:
    - directory: Path to the directory containing .tif files.

    Prints the result of the checks for each .tif file.
    """
    # List all .tif files in the directory
    invalid = []
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            file_path = os.path.join(directory, filename)
            try:
                # Open the .tif file with rasterio
                with rasterio.open(file_path) as src:
                    # Check the number of bands
                    band_count = src.count
                    if band_count != 3:
                        print(
                            f"Warning: {filename} contains {band_count} bands (should be 3)"
                        )
                        invalid.append(filename)
                        continue

                    # Check if there is valid data in all 3 bands
                    invalid_bands = False
                    for i in range(1, 4):  # Loop through bands 1, 2, 3
                        data = src.read(i)
                        if not data.any():
                            invalid_bands = True
                    if invalid_bands:
                        print(f"Warning: Bands of {filename} contains no valid data")
                        invalid.append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                invalid.append(filename)
    return invalid


# Warning: lit_41.61618_-124.11507.json does not contain 'mapData'.
# Warning: lit_37.59522_-122.52383.json does not contain 'mapData'.
# Warning: lit_41.14219_-124.15947.json does not contain 'mapData'.
# Warning: lit_41.5072_-71.0885.json does not contain 'mapData'.
# Warning: lit_46.58611_-87.37897.json does not contain 'mapData'.
# Warning: lit_42.26058_-70.88709.json does not contain 'mapData'.
# Warning: lit_43.84139_-69.50198.json does not contain 'mapData'.
# Warning: lit_35.8558_-121.4153.json does not contain 'mapData'.
def check_lithology_for_map_data(directory):
    """
    Check each .json file in the given directory to ensure it contains 'mapData'.

    Parameters:
    - directory: Path to the directory containing .json files.

    Prints the result of the checks for each .json file.
    """
    invalid = []
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                # Open and load the JSON file
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)

                    # Check if 'mapData' exists in the correct structure
                    if not (
                        "success" in data
                        and "data" in data["success"]
                        and "mapData" in data["success"]["data"]
                        and len(data["success"]["data"]["mapData"]) >= 1
                    ):
                        print(f"Warning: {filename} does not contain 'mapData'.")
                        invalid.append(filename)
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
                invalid.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                invalid.append(filename)
    return invalid


# Confirmed
def check_parallelism():
    dir1 = "data/labaled/dem"
    dir2 = "data/labaled/dem_parallel"
    mismatched_files = compare_tif_files(dir1, dir2)
    # Return the list of mismatched files
    if mismatched_files:
        print(f"Mismatched files: {mismatched_files}")
    else:
        print("All files match.")

    dir1 = "data/labeled/sentinel2"
    dir2 = "data/labeled/sentinel2_parallel"
    mismatched_files = compare_tif_files(dir1, dir2)
    # Return the list of mismatched files
    if mismatched_files:
        print(f"Mismatched files: {mismatched_files}")
    else:
        print("All files match.")


# Confirmed
def check_same_files():
    dir1 = "./data/labeled/dem"
    dir2 = "./data/labeled/sentinel2"
    dir3 = "./data/labeled/lithology"
    compare_directories(dir1, dir2, dir3)


def remove_invalids(file_names):
    for f in file_names:
        lat, lon = decode_file(f)
        dem_file = encode_file(lat, lon, "dem", file_type="tif")
        delete_file(dem_file, "./data/labeled/dem")
        sen_file = encode_file(lat, lon, "sen", file_type="tif")
        delete_file(sen_file, "./data/labeled/sentinel2")
        lit_file = encode_file(lat, lon, "lit", file_type="json")
        delete_file(lit_file, "./data/labeled/lithology")


if __name__ == "__main__":
    # check_parallelism()
    check_same_files()
    invalid_dems = check_dem_files("./data/labeled/dem")
    invalid_sen = check_sen_files("./data/labeled/sentinel2")
    invalid_lith = check_lithology_for_map_data("./data/labeled/lithology")
    invalids = invalid_dems + invalid_sen + invalid_lith
    print(invalids)
    # invalids = [
    #     "dem_37.59522_-122.52383.tif",
    #     "dem_42.26058_-70.88709.tif",
    #     "sen_43.99828_-71.19971.tif",
    #     "sen_35.86384_-93.04812.tif",
    #     "sen_34.21976_-116.9928.tif",
    #     "sen_34.27148_-116.98311.tif",
    #     "sen_34.21994_-116.99222.tif",
    #     "sen_44.0447_-71.3618.tif",
    #     "sen_34.27822_-116.9698.tif",
    #     "sen_44.00694_-71.23067.tif",
    #     "sen_34.21992_-116.99268.tif",
    #     "sen_44.00669_-71.23158.tif",
    #     "sen_44.03984_-71.39582.tif",
    #     "sen_43.99824_-71.20047.tif",
    #     "sen_43.99821_-71.20059.tif",
    #     "sen_44.01938_-71.25702.tif",
    #     "sen_43.99844_-71.19998.tif",
    #     "sen_44.00839_-71.23031.tif",
    #     "sen_44.043_-71.3591.tif",
    #     "sen_43.9986_-71.19948.tif",
    #     "sen_44.01767_-71.26066.tif",
    #     "sen_44.00903_-71.2253.tif",
    #     "sen_44.02081_-71.25632.tif",
    #     "sen_35.84203_-93.05528.tif",
    #     "sen_34.2276_-116.9883.tif",
    #     "sen_35.8416_-93.0534.tif",
    #     "sen_44.02013_-71.25665.tif",
    #     "sen_34.21983_-116.99266.tif",
    #     "sen_44.04497_-71.36221.tif",
    #     "sen_34.21967_-116.99275.tif",
    #     "lit_41.61618_-124.11507.json",
    #     "lit_37.59522_-122.52383.json",
    #     "lit_41.14219_-124.15947.json",
    #     "lit_41.5072_-71.0885.json",
    #     "lit_46.58611_-87.37897.json",
    #     "lit_42.26058_-70.88709.json",
    #     "lit_43.84139_-69.50198.json",
    #     "lit_35.8558_-121.4153.json",
    # ]
    # remove_invalids(invalids)
