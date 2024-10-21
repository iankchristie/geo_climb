import pandas as pd
from sentinel2_adapter import *
from file_utils import *
import concurrent.futures
import time


def sentinel2_download(
    data_dir: str = "data/sentinel2",
    limit: int | None = 10,
):
    lat_lons = get_undownloaded_lat_lons(data_dir, limit)
    sentinel2_downloader = Sentinel2Adapter(output_folder=data_dir)

    start_time = time.time()
    for lat, lon in lat_lons:
        sentinel2_downloader.download(latitude=lat, longitude=lon)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Synchronous: Finished downloading {len(lat_lons)} SENTINEL2 files in {elapsed_time:.2f} seconds."
    )


# NOTE: To use these parallel methods you MUST ensure that your downloader is Thread Safe!
def sentinel2_download_parallel(
    data_dir: str = "data/sentinel2_parallel", limit: int | None = 10
):
    lat_lons = get_undownloaded_lat_lons(data_dir, limit)
    sentinel2_downloader = Sentinel2Adapter(output_folder=data_dir)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all the download tasks to run concurrently
        futures = [
            executor.submit(
                sentinel2_downloader.download,
                latitude=lat,
                longitude=lon,
            )
            for lat, lon in lat_lons
        ]

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during download: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Parallel: Finished downloading {len(lat_lons)} SENTINEL2 files in {elapsed_time:.2f} seconds."
    )


if __name__ == "__main__":
    # sentinel2_download(limit=10)
    sentinel2_download_parallel(limit=10)
