import rasterio
from matplotlib import pyplot as plt
import os


def view_dem(
    latitude: float = 40.0150,
    longitude: float = -105.2705,
    output_folder: str = "data/dem",
):
    filename_base = f"dem_{latitude}_{longitude}"
    output_tif = os.path.join(output_folder, f"{filename_base}.tif")

    with rasterio.open(output_tif) as src:
        dem_data = src.read(1)

    # Plot the DEM data
    plt.imshow(dem_data, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM")
    plt.show()


if __name__ == "__main__":
    view_dem(latitude=40.0150, longitude=-105.2705)
