import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from sentinel2_viewer import *
from dem_viewer import *
from lithology_viewer import *
from downloaders.file_utils import *
from config.config import Config


def plot_data(lat: float, lon: float):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    view_sentinel2(
        data_dir=Config.DATA_DIR_UNLBL_SEN,
        latitude=lat,
        longitude=lon,
        ax=axs[0]
    )
    im = view_dem(data_dir=Config.DATA_DIR_UNLBL_DEM, latitude=lat, longitude=lon, ax=axs[1])
    fig.colorbar(im, ax=axs[1], label="Elevation (m)")
    view_lithology(
        data_dir=Config.DATA_DIR_UNLBL_LITH, latitude=lat, longitude=lon, ax=axs[2]
    )
    fig.suptitle(f"Latitude: {lat}, Longitude: {lon}", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dem_points = get_lat_lons_from_directory(Config.DATA_DIR_UNLBL_DEM)
    sen_points = get_lat_lons_from_directory(Config.DATA_DIR_UNLBL_SEN)
    lit_points = get_lat_lons_from_directory(Config.DATA_DIR_UNLBL_LITH)
    downloaded_points = dem_points.intersection(sen_points).intersection(lit_points)
    for lat, lon in downloaded_points:
        plot_data(lat, lon)
