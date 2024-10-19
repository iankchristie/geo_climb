import ee
import requests
import zipfile
import io
import rasterio
import numpy as np
from matplotlib import pyplot as plt

# ee.Authenticate()
# ee.Initialize(project="geospatialml")

# latitude = 40.0150
# longitude = -105.2705

# point = ee.Geometry.Point([longitude, latitude])

# sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR")

# sentinel2_filtered = (
#     sentinel2.filterBounds(point)
#     .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 10)
#     .select(["B2", "B3", "B4"])  # Select only the RGB bands
# )  #

# image = sentinel2_filtered.sort("system:time_start", False).first()

# if image:
#     # Define the region to download (e.g., a 1km buffer around the point)
#     region = point.buffer(1000).bounds()

#     # Clip the image to the region of interest
#     clipped_image = image.clip(region)

#     # Get the download URL
#     url = clipped_image.getDownloadURL(
#         {
#             "scale": 10,  # Sentinel-2 resolution is 10 meters
#             "crs": "EPSG:4326",
#             "region": region.getInfo()["coordinates"],
#         }
#     )

#     # Download the image
#     response = requests.get(url, stream=True)

#     if response.status_code == 200:
#         # Save and extract the ZIP file containing the image bands
#         with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#             z.extractall("sentinel_image")
#         print("Image downloaded and extracted to sentinel_image folder.")
#     else:
#         print("Failed to download image.")
# else:
#     print("No image found for the specified location and criteria.")

# Open the bands
with rasterio.open(
    "sentinel_image/20241007T175129_20241007T175718_T13TDE.B4.tif"
) as red_band:
    red = red_band.read(1)

with rasterio.open(
    "sentinel_image/20241007T175129_20241007T175718_T13TDE.B3.tif"
) as green_band:
    green = green_band.read(1)

with rasterio.open(
    "sentinel_image/20241007T175129_20241007T175718_T13TDE.B2.tif"
) as blue_band:
    blue = blue_band.read(1)

# Stack the bands into a 3D array (R, G, B)
rgb = np.dstack((red, green, blue))

# Display the image
plt.imshow(rgb / np.max(rgb))  # Normalize values to 0-1 for display
plt.title("True Color Composite (RGB)")
plt.show()
