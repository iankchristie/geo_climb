import ee
import requests
import zipfile
import io
import os
import rasterio
from glob import glob
from filelock import FileLock


class Sentinel2Adapter:
    def __init__(
        self,
        output_folder: str = "data/sentinel2",
    ) -> None:
        # Authenticate and initialize Earth Engine
        ee.Authenticate()
        ee.Initialize(project="geospatialml")
        self.output_folder = output_folder
        self.lock_file = os.path.join(self.output_folder, "process.lock")

    def download(self, latitude: float = 40.0150, longitude: float = -105.2705):
        point = ee.Geometry.Point([longitude, latitude])

        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR")

        sentinel2_filtered = (
            sentinel2.filterBounds(point)
            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 10)
            .select(["B2", "B3", "B4"])  # Select only the RGB bands
        )

        image = sentinel2_filtered.sort("system:time_start", False).first()

        if image:
            # Define the region to download (e.g., a 1km buffer around the point)
            region = point.buffer(1000).bounds()

            # Clip the image to the region of interest
            clipped_image = image.clip(region)

            # Get the download URL
            url = clipped_image.getDownloadURL(
                {
                    "scale": 10,  # Sentinel-2 resolution is 10 meters
                    "crs": "EPSG:4326",
                    "region": region.getInfo()["coordinates"],
                }
            )

            # Download the image
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with FileLock(self.lock_file, timeout=20):
                    # Save and extract the ZIP file containing the image bands
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        z.extractall(self.output_folder)

                    print("Image downloaded and extracted.")

                    # Stack the bands into a single multi-band GeoTIFF
                    self._stack_bands_into_tiff(latitude, longitude)
            else:
                print("Failed to download image.")
        else:
            print("No image found for the specified location and criteria.")

    def _stack_bands_into_tiff(self, latitude: float, longitude: float):
        """
        Combine the downloaded RGB bands into a single GeoTIFF file.
        """
        # Generate the output filename based on latitude and longitude
        output_tif = os.path.join(self.output_folder, f"sen_{latitude}_{longitude}.tif")

        # Use glob to find the actual file paths for B2, B3, B4 (with timestamps in names)
        band_files = {"B2": None, "B3": None, "B4": None}

        for band in band_files:
            # Find the file that ends with the band name (e.g., B2.tif, B3.tif, B4.tif)
            files = glob(os.path.join(self.output_folder, f"*{band}.tif"))
            if files:
                band_files[band] = files[0]  # Take the first match

        if None in band_files.values():
            print("Error: One or more band files could not be found.")
            return

        # Stack the bands into a single multi-band GeoTIFF
        band_data = []
        profile = None

        for band_file in band_files.values():
            with rasterio.open(band_file) as src:
                band_data.append(src.read(1))  # Read the first band
                if profile is None:
                    profile = src.profile  # Capture the profile from the first band

        # Update the profile to reflect a multi-band image
        profile.update(count=len(band_data))

        # Write the stacked bands into a single GeoTIFF
        with rasterio.open(output_tif, "w", **profile) as dst:
            for idx, band in enumerate(band_data, start=1):
                dst.write(band, idx)  # Write each band to the corresponding index

        print(f"Stacked RGB bands saved as {output_tif}")

        # Delete the original band files
        self._delete_band_files(band_files)

    def _delete_band_files(self, band_files: dict):
        """
        Delete the original individual band files (B2, B3, B4) after they have been combined.
        """
        for band, file_path in band_files.items():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            else:
                print(f"File {file_path} does not exist.")


if __name__ == "__main__":
    sen_adapter = Sentinel2Adapter()
    sen_adapter.download(latitude=40.0150, longitude=-105.2705)
