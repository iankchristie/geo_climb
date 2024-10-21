import rasterio
from matplotlib import pyplot as plt
import os
from file_utils import *


def view_dem(
    latitude: float = 40.0150,
    longitude: float = -105.2705,
    data_dir: str = "data/dem",
):
    output_tif = encode_file(latitude, longitude, "dem", data_dir)

    with rasterio.open(output_tif) as src:
        dem_data = src.read(1)

    # Plot the DEM data
    plt.imshow(dem_data, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM")
    plt.show()


if __name__ == "__main__":
    view_dem(latitude=40.0150, longitude=-105.2705)
