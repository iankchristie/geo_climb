from dem_adapter import *
from sentinel2_adapter import *
from lithology_adapter import *
from file_utils import *
import concurrent.futures
import time
from adapter import *


# NOTE: To use the parallel functionality you MUST ensure that your downloader is Thread Safe!
def download(adapter: SafeAdapter, limit: int | None = 10, parallel: bool = False):
    data_dir = adapter.output_folder
    lat_lons = get_undownloaded_lat_lons(data_dir, limit)

    start_time = time.time()
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all the download tasks to run concurrently
            futures = [
                executor.submit(
                    adapter.download,
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
    else:
        for lat, lon in lat_lons:
            adapter.download(latitude=lat, longitude=lon)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Finished downloading {len(lat_lons)} DEM files in {elapsed_time:.2f} seconds."
    )


if __name__ == "__main__":
    # download(DEMAdapter(output_folder="data/dem"), limit=50)
    # download(DEMAdapter(output_folder="data/dem_parallel"), limit=50, parallel=True)
    # download(Sentinel2Adapter(output_folder="data/sentinel2"), limit=50)
    # download(
    #     Sentinel2Adapter(output_folder="data/sentinel2_parallel"),
    #     limit=50,
    #     parallel=True,
    # )
    download(LithologyAdapter(output_folder="data/lithology"), limit=50)
    download(
        LithologyAdapter(output_folder="data/lithology_parallel"),
        limit=50,
        parallel=True,
    )
