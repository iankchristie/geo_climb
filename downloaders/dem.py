import ee
import requests
import zipfile
import io
import rasterio
from matplotlib import pyplot as plt


# Authenticate and initialize Earth Engine
# ee.Authenticate()
# ee.Initialize(project="geospatialml")

# latitude = 40.0150
# longitude = -105.2705

# # Create a point geometry
# point = ee.Geometry.Point([longitude, latitude])

# # Get the SRTM DEM ImageCollection
# dem = ee.Image("USGS/SRTMGL1_003")  # SRTM Global 1 arc-second DEM

# # Define the region to download (e.g., a 1km buffer around the point)
# region = point.buffer(1000).bounds()

# # Clip the DEM to the region of interest
# clipped_dem = dem.clip(region)

# # Get the download URL for the DEM data
# url = clipped_dem.getDownloadURL(
#     {
#         "scale": 30,  # SRTM resolution is 30 meters
#         "crs": "EPSG:4326",
#         "region": region.getInfo()["coordinates"],
#     }
# )

# # Download the DEM data
# response = requests.get(url, stream=True)

# if response.status_code == 200:
#     # Save and extract the ZIP file containing the DEM data
#     with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#         z.extractall("data/dem")
#     print("DEM data downloaded and extracted to dem folder.")
# else:
#     print("Failed to download DEM data.")

# data/dem/SRTMGL1_003.elevation.tif
with rasterio.open("data/dem/SRTMGL1_003.elevation.tif") as src:
    dem_data = src.read(1)

# Plot the DEM data
plt.imshow(dem_data, cmap="terrain")
plt.colorbar(label="Elevation (m)")
plt.title("DEM")
plt.show()
