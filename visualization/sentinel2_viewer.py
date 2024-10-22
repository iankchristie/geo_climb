import rasterio
import numpy as np
from matplotlib import pyplot as plt
import os


def view_sentinel2(
    latitude: float = 40.0150,
    longitude: float = -105.2705,
    data_dir: str = "data/sentinel2",
):
    tif_file = os.path.join(data_dir, f"sen_{latitude}_{longitude}.tif")

    # Open the multi-band GeoTIFF file
    with rasterio.open(tif_file) as src:
        # Read the bands (assuming B4 is Red, B3 is Green, B2 is Blue)
        red = src.read(1)  # Band 1 (B4: Red)
        green = src.read(2)  # Band 2 (B3: Green)
        blue = src.read(3)  # Band 3 (B2: Blue)

    # Stack the bands into a 3D array (R, G, B)
    rgb = np.dstack((red, green, blue))

    # Display the image
    plt.imshow(rgb / np.max(rgb))  # Normalize values to 0-1 for display
    plt.title(f"True Color Composite (RGB) for {latitude}, {longitude}")
    plt.show()


if __name__ == "__main__":
    # data/sentinel2/sen_37.51637_-118.57089.tif
    view_sentinel2(latitude=37.51637, longitude=-118.57089)
