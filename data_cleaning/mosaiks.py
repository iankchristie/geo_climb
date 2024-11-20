import os
import sys
import pandas as pd
import pdb
from geopy.distance import geodesic
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lat_lon_tree import LatLonTree
from utils.file_utils import *

# Load the CSV files into DataFrames
all_lat_lons = pd.read_csv("data/all_lat_lons.csv")
sentinel_mosaiks = pd.read_csv("data/sentinel_mosaiks.csv")

# Ensure lat-lon columns are named consistently
all_lat_lons.columns = all_lat_lons.columns.str.strip()
sentinel_mosaiks.columns = sentinel_mosaiks.columns.str.strip()

# Create sets of lat-lon tuples from both DataFrames
all_lat_lon_set = set(
    zip(all_lat_lons["latitude"], all_lat_lons["longitude"], all_lat_lons["labeled"])
)
sentinel_mosaiks_set = set(zip(sentinel_mosaiks["Lat"], sentinel_mosaiks["Lon"]))
lat_lon_tree = LatLonTree(list(sentinel_mosaiks_set))


# Function to calculate the distance to the nearest point
def nearest_distance(point, reference_points):
    return min(geodesic(point, ref_point).kilometers for ref_point in reference_points)


# Calculate nearest distance for each point in all_lat_lon_set
results = []
labeled_directory = "data/labeled/embeddings/sentinel_mosaiks"
os.makedirs(labeled_directory, exist_ok=True)
unlabeled_directory = "data/unlabeled/embeddings/sentinel_mosaiks"
os.makedirs(unlabeled_directory, exist_ok=True)


for lat, lon, label in all_lat_lon_set:
    distance, nearest_lat_lon = lat_lon_tree.query((lat, lon))
    # Find the row in sentinel_mosaiks matching the nearest lat-lon
    nearest_row = sentinel_mosaiks[
        (sentinel_mosaiks["Lat"] == nearest_lat_lon[0])
        & (sentinel_mosaiks["Lon"] == nearest_lat_lon[1])
    ]
    # Add the original point, distance, nearest lat-lon, and the rest of the columns from sentinel_mosaiks
    if not nearest_row.empty:
        if label:
            directory = labeled_directory
        else:
            directory = unlabeled_directory
        file_path = encode_file(lat, lon, "sen", directory, "npy")
        # Skips the lat, lon
        nearest_data = nearest_row.iloc[0, 2:].to_numpy()
        np.save(file_path, nearest_data)
