import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from sentinel2_viewer import *
from dem_viewer import *
from downloaders.file_utils import *


def plot_data(lat: float, lon: float):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    view_sentinel2(latitude=lat, longitude=lon, ax=axs[0])
    im = view_dem(latitude=lat, longitude=lon, ax=axs[1])
    fig.colorbar(im, ax=axs[1], label="Elevation (m)")
    fig.suptitle(f"Data for Latitude: {lat}, Longitude: {lon}", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dem_points = get_lat_lons_from_directory("data/dem")
    sen_points = get_lat_lons_from_directory("data/sentinel2")
    downloaded_points = dem_points and sen_points
    for lat, lon in downloaded_points:
        plot_data(lat, lon)
