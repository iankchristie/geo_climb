import os
import sys
import glob
import numpy as np

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import *


sen_directory = "sentinel_mosaiks"
dem_directory = "dem_rcf_empirical"
lithology_directory = "lithology_scibert_no_description"

labeled_directory = "data/unlabeled/embeddings"
unlabeled_directory = "data/unlabeled/embeddings"

for directory in [labeled_directory, unlabeled_directory]:
    sen_paths = []
    dem_paths = []
    lith_paths = []

    data_types = [
        (sen_directory, sen_paths),
        (dem_directory, dem_paths),
        (lithology_directory, lith_paths),
    ]

    for data_directory, agg in data_types:
        full_directory = os.path.join(directory, data_directory)
        for filename in glob.glob(os.path.join(full_directory, "*.npy")):
            agg.append(filename)
    sen_paths.sort()
    dem_paths.sort()
    lith_paths.sort()

    for i in range(len(sen_paths)):
        if decode_file(sen_paths[i]) != decode_file(dem_paths[i]) or decode_file(
            sen_paths[i]
        ) != decode_file(lith_paths[i]):
            print("Oh no!")

    combined_directory = os.path.join(directory, "combined")
    os.makedirs(combined_directory, exist_ok=True)

    for i in range(len(sen_paths)):
        sen_data = np.load(sen_paths[i])
        dem_data = np.load(dem_paths[i])
        lith_data = np.load(lith_paths[i])

        combined_data = np.concatenate((sen_data, dem_data, lith_data))

        combined_file_name = os.path.basename(sen_paths[i]).replace("sen", "com")
        full_combined_path = os.path.join(combined_directory, combined_file_name)
        # print(full_combined_path)
        np.save(full_combined_path, combined_data)
