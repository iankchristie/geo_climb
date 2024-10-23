import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.axes import Axes
import rasterio
from downloaders.file_utils import *


def view_dem(
    data_dir: str,
    ax: Axes,
    latitude: float = 40.0150,
    longitude: float = -105.2705,
):
    output_tif = encode_file(latitude, longitude, "dem", data_dir)

    with rasterio.open(output_tif) as src:
        dem_data = src.read(1)

    print(f"dem shape: {dem_data.shape}")
    im = ax.imshow(dem_data, cmap="terrain")
    ax.axis("off")

    # Return the colorbar to the caller for adding it to the main figure
    return im
