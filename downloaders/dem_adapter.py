import ee
import requests
import zipfile
import io
import os
import threading


file_lock = threading.Lock()


class DEMAdapter:
    def __init__(
        self,
        output_folder: str = "data/dem",
    ) -> None:
        # Authenticate and initialize Earth Engine
        ee.Authenticate()
        ee.Initialize(project="geospatialml")
        self.output_folder = output_folder

    # Defaulting to Boulder to make testing easier
    def download(self, latitude: float = 40.0150, longitude: float = -105.2705):
        # Create a point geometry
        point = ee.Geometry.Point([longitude, latitude])

        # Get the SRTM DEM ImageCollection
        dem = ee.Image("USGS/SRTMGL1_003")  # SRTM Global 1 arc-second DEM

        # Define the region to download (e.g., a 1km buffer around the point)
        region = point.buffer(1000).bounds()

        # Clip the DEM to the region of interest
        clipped_dem = dem.clip(region)

        # Get the download URL for the DEM data
        url = clipped_dem.getDownloadURL(
            {
                "scale": 30,  # SRTM resolution is 30 meters
                "crs": "EPSG:4326",
                "region": region.getInfo()["coordinates"],
            }
        )

        # Create a base filename based on latitude and longitude
        filename_base = f"dem_{latitude}_{longitude}"
        output_tif = os.path.join(self.output_folder, f"{filename_base}.tif")

        # Download the ZIP file containing the DEM data
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            # Ensure the output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with file_lock:  # Ensure exclusive access to file operations
                    for file_info in z.infolist():
                        if file_info.filename.endswith(".tif"):
                            extracted_tif = os.path.join(
                                self.output_folder, file_info.filename
                            )
                            z.extract(file_info, self.output_folder)

                            # Rename the file using latitude and longitude
                            os.rename(extracted_tif, output_tif)
                            print(f"DEM data extracted and saved as {output_tif}")
                            return

            print(
                f"No .tif file found in the downloaded ZIP for coordinates ({latitude}, {longitude})."
            )
        else:
            print(
                f"Failed to download DEM data for coordinates ({latitude}, {longitude})."
            )


if __name__ == "__main__":
    dem_downloader = DEMAdapter()
    dem_downloader.download(latitude=40.0150, longitude=-105.2705)
