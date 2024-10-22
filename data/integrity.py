import os
import rasterio
import numpy as np


def compare_tif_files(dir1: str, dir2: str):
    """
    Compare .tif files in two directories that share the same file name.
    Checks if the contents of the files (e.g., pixel values, metadata) are equal.

    Parameters:
    - dir1: Path to the first directory.
    - dir2: Path to the second directory.

    Returns:
    - A list of files that are not identical between the two directories.
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


# Example usage
if __name__ == "__main__":
    dir1 = "data/dem"
    dir2 = "data/dem_parallel"
    mismatched_files = compare_tif_files(dir1, dir2)
    # Return the list of mismatched files
    if mismatched_files:
        print(f"Mismatched files: {mismatched_files}")
    else:
        print("All files match.")

    dir1 = "data/sentinel2"
    dir2 = "data/sentinel2_parallel"
    mismatched_files = compare_tif_files(dir1, dir2)
    # Return the list of mismatched files
    if mismatched_files:
        print(f"Mismatched files: {mismatched_files}")
    else:
        print("All files match.")
