import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.axes import Axes
import rasterio
from downloaders.file_utils import *


def view_dem(
    ax: Axes,
    latitude: float = 40.0150,
    longitude: float = -105.2705,
    data_dir: str = "data/dem",
):
    """
    Plots the DEM data for the given latitude and longitude on the provided axes.
    """
    output_tif = encode_file(latitude, longitude, "dem", data_dir)

    with rasterio.open(output_tif) as src:
        dem_data = src.read(1)

    im = ax.imshow(dem_data, cmap="terrain")
    ax.set_title(f"DEM")
    ax.axis("off")

    # Return the colorbar to the caller for adding it to the main figure
    return im
