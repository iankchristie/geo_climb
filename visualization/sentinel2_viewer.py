import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib.axes import Axes
import rasterio
import numpy as np
from downloaders.file_utils import *


def view_sentinel2(
    data_dir: str,
    ax: Axes,
    latitude: float = 40.0150,
    longitude: float = -105.2705,
):
    tif_file = os.path.join(data_dir, f"sen_{latitude}_{longitude}.tif")

    # Open the multi-band GeoTIFF file
    with rasterio.open(tif_file) as src:
        blue = src.read(1)  # Band 1 (B2: Blue)
        green = src.read(2)  # Band 2 (B3: Green)
        red = src.read(3)  # Band 3 (B4: Red)

    # Stack the bands into a 3D array (R, G, B)
    rgb = np.dstack((red, green, blue))

    print(f"sen shape: {rgb.shape}")
    ax.imshow(rgb / np.max(rgb))  # Normalize values to 0-1 for display
    ax.set_title(f"True Color Composite (RGB)")
    ax.axis("off")
