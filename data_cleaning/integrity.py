import json
import os
import sys
import rasterio
import numpy as np

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import *


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
def check_same_files_labeled():
    dir1 = "./data/labeled/dem"
    dir2 = "./data/labeled/sentinel2"
    dir3 = "./data/labeled/lithology"
    compare_directories(dir1, dir2, dir3)


def check_same_files_unlabeled():
    dir1 = "./data/unlabeled/dem"
    dir2 = "./data/unlabeled/sentinel2"
    dir3 = "./data/unlabeled/lithology"
    compare_directories(dir1, dir2, dir3)


def remove_labeled_invalids(file_names):
    for f in file_names:
        lat, lon = decode_file(f)
        dem_file = encode_file(lat, lon, "dem", file_type="tif")
        delete_file(dem_file, "./data/labeled/dem")
        sen_file = encode_file(lat, lon, "sen", file_type="tif")
        delete_file(sen_file, "./data/labeled/sentinel2")
        lit_file = encode_file(lat, lon, "lit", file_type="json")
        delete_file(lit_file, "./data/labeled/lithology")


def remove_unlabeled_invalids(file_names):
    for f in file_names:
        lat, lon = decode_file(f)
        dem_file = encode_file(lat, lon, "dem", file_type="tif")
        delete_file(dem_file, "./data/unlabeled/dem")
        sen_file = encode_file(lat, lon, "sen", file_type="tif")
        delete_file(sen_file, "./data/unlabeled/sentinel2")
        lit_file = encode_file(lat, lon, "lit", file_type="json")
        delete_file(lit_file, "./data/unlabeled/lithology")


if __name__ == "__main__":
    # check_parallelism()
    # check_same_files_labeled()
    # invalid_dems = check_dem_files("./data/labeled/dem")
    # invalid_sen = check_sen_files("./data/labeled/sentinel2")
    # invalid_lith = check_lithology_for_map_data("./data/labeled/lithology")
    # invalids = invalid_dems + invalid_sen + invalid_lith
    # print(invalids)
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
    # remove_labeled_invalids(invalids)

    # check_same_files_unlabeled()
    # invalid_dems = check_dem_files("./data/unlabeled/dem")
    # invalid_sen = check_sen_files("./data/unlabeled/sentinel2")
    # invalid_lith = check_lithology_for_map_data("./data/unlabeled/lithology")
    # invalids = invalid_dems + invalid_sen + invalid_lith
    invalids = [
        "dem_47.87591831569789_-122.4797336993544.tif",
        "dem_29.167147557841393_-90.58098179696498.tif",
        "dem_29.799719577302135_-94.81070117793438.tif",
        "dem_27.253627001548864_-97.5002800758797.tif",
        "dem_35.133066340222435_-76.20057530670462.tif",
        "dem_28.461992982813868_-96.38270471047063.tif",
        "dem_29.80981586242405_-91.96698469327517.tif",
        "dem_29.77783737098401_-85.39663912639989.tif",
        "dem_35.006199519988485_-76.4058881572184.tif",
        "dem_43.78951146615729_-70.03520749625167.tif",
        "dem_29.880747823262983_-92.76335970989405.tif",
        "dem_28.693882917782847_-96.6176823884899.tif",
        "dem_30.234505117233176_-90.27576403766182.tif",
        "dem_28.647007745719815_-96.02764853338353.tif",
        "dem_29.07713263779803_-90.72833572427535.tif",
        "dem_48.62240835622018_-123.19166212513684.tif",
        "dem_24.9353052720035_-80.69083508207294.tif",
        "dem_29.92020287869038_-93.84393183027952.tif",
        "dem_47.64285133898961_-122.4313190968358.tif",
        "dem_30.069932040474377_-81.68079050719498.tif",
        "dem_37.578593905191205_-76.33012759102957.tif",
        "dem_29.964568460805967_-89.24440001341085.tif",
        "dem_35.19253394814419_-76.22643517042482.tif",
        "dem_34.82095623158587_-76.38627647359829.tif",
        "dem_44.32909043507138_-68.4707737271757.tif",
        "dem_26.138370922344514_-97.19917166618517.tif",
        "dem_24.569340347045262_-82.12940829459072.tif",
        "dem_35.433503402009244_-75.62744358685805.tif",
        "dem_37.22516639305809_-76.8056944256071.tif",
        "dem_31.733133911623703_-81.20369093924475.tif",
        "dem_37.37236616747507_-75.7582898306851.tif",
        "dem_30.042501347521128_-90.16824446728356.tif",
        "dem_34.8617507350386_-76.35686241249245.tif",
        "dem_35.03314143246928_-76.93990757504754.tif",
        "dem_41.637049341783_-70.72795729294494.tif",
        "dem_28.274205178368277_-96.78826455243777.tif",
        "dem_28.20110122112307_-82.80477510828507.tif",
        "dem_44.436568306654074_-124.038167709073.tif",
        "dem_35.323796236987505_-75.9759748004661.tif",
        "dem_30.1138791167448_-90.24645569970642.tif",
        "dem_26.639418970222888_-82.24383636910717.tif",
        "dem_47.40359238188251_-122.40138753260392.tif",
        "dem_30.298882814167943_-81.68600319661445.tif",
        "dem_29.633365883002504_-85.03972255898805.tif",
        "dem_41.6491198840691_-70.7246673413502.tif",
        "sen_30.335816449449716_-92.84179204494784.tif",
        "sen_45.74572122790403_-113.48626638288476.tif",
        "sen_34.93383176748379_-89.11084988756829.tif",
        "sen_31.327793160482013_-102.65256840782342.tif",
        "sen_43.64020239953332_-100.34222965711103.tif",
        "sen_33.95306117869469_-91.89159862185488.tif",
        "sen_37.12563366945852_-80.0393492770149.tif",
        "sen_41.09823097970258_-77.25577334836393.tif",
        "sen_37.552221327004304_-78.30837557466452.tif",
        "sen_38.38187333412743_-90.64945330958814.tif",
        "sen_33.264986497842685_-84.5181943842556.tif",
        "lit_35.74286198506596_-76.04910336372495.json",
        "lit_34.82095623158587_-76.38627647359829.json",
        "lit_26.639418970222888_-82.24383636910717.json",
        "lit_44.30666420767984_-76.05854480796242.json",
        "lit_29.44729462378916_-91.31045691394702.json",
        "lit_45.98504409750684_-84.6437024940422.json",
        "lit_29.205831135096872_-90.04345832201437.json",
        "lit_29.504838151323213_-90.06026783521466.json",
        "lit_45.94854653499037_-84.64685645133628.json",
        "lit_36.15387426161901_-76.38743443876174.json",
        "lit_29.167147557841393_-90.58098179696498.json",
        "lit_25.6049296818408_-81.24129493190077.json",
        "lit_39.180405346441304_-77.5153313260976.json",
        "lit_45.97343526131601_-85.82360926604056.json",
        "lit_25.89191551545033_-81.52427777079215.json",
        "lit_34.8617507350386_-76.35686241249245.json",
        "lit_37.22516639305809_-76.8056944256071.json",
        "lit_26.46308235447037_-82.05538527082898.json",
        "lit_44.43162778071148_-75.77776262886063.json",
        "lit_29.63883493738222_-91.57375156274978.json",
        "lit_42.52257049381869_-82.86962495398501.json",
        "lit_29.07713263779803_-90.72833572427535.json",
        "lit_28.310486459644828_-80.63118682949522.json",
        "lit_25.263734301943607_-80.30111954912212.json",
        "lit_29.80981586242405_-91.96698469327517.json",
        "lit_29.964568460805967_-89.24440001341085.json",
        "lit_28.741754882085452_-82.66550135085583.json",
        "lit_29.29221986199801_-90.54484001890698.json",
        "lit_35.19253394814419_-76.22643517042482.json",
        "lit_35.323796236987505_-75.9759748004661.json",
        "lit_30.1138791167448_-90.24645569970642.json",
        "lit_24.9353052720035_-80.69083508207294.json",
        "lit_29.633365883002504_-85.03972255898805.json",
        "lit_37.37236616747507_-75.7582898306851.json",
        "lit_43.876733948610536_-83.41145121312823.json",
        "lit_25.152172501614796_-80.71686500636393.json",
        "lit_29.77783737098401_-85.39663912639989.json",
        "lit_28.56246140280807_-80.60630633826692.json",
        "lit_30.042501347521128_-90.16824446728356.json",
        "lit_28.20110122112307_-82.80477510828507.json",
        "lit_35.006199519988485_-76.4058881572184.json",
        "lit_43.65154900598158_-83.8196868975471.json",
        "lit_38.08061870211619_-76.96923753870315.json",
        "lit_26.652334106649043_-82.18361401318515.json",
        "lit_29.90348520747973_-81.2940190477529.json",
        "lit_41.637049341783_-70.72795729294494.json",
        "lit_30.234505117233176_-90.27576403766182.json",
        "lit_41.49234478024574_-82.77296215301796.json",
        "lit_35.133066340222435_-76.20057530670462.json",
        "lit_35.03314143246928_-76.93990757504754.json",
        "lit_35.433503402009244_-75.62744358685805.json",
        "lit_41.6491198840691_-70.7246673413502.json",
        "lit_24.569340347045262_-82.12940829459072.json",
        "lit_37.578593905191205_-76.33012759102957.json",
        "lit_46.2115987124381_-84.14034568456537.json",
        "lit_44.99892838845745_-85.40089723997944.json",
    ]
    print(len(invalids))
    remove_unlabeled_invalids(invalids)
