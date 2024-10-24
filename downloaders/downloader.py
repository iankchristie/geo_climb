from dem_adapter import *
from sentinel2_adapter import *
from lithology_adapter import *
from file_utils import *
import concurrent.futures
import time
from safe_adapter import *


# NOTE: To use the parallel functionality you MUST ensure that your downloader is Thread Safe!
def download(
    locations_file: str,
    adapter: SafeAdapter,
    limit: int | None,
    parallel: bool = False,
):
    data_dir = adapter.output_folder
    lat_lons = get_undownloaded_lat_lons(locations_file, data_dir, limit)

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


# Instructions: Uncomment and run the datasets you wish to download. The parallel functionality
# greatly speeds up download, but please do not run the lithology download in parallel, it could
# overwhelm the macrostrat servers.
if __name__ == "__main__":
    # download(
    #     locations_file="data/labeled/climbing_locations.csv",
    #     adapter=DEMAdapter(output_folder="data/labeled/dem"),
    #     parallel=True,
    #     limit=None,
    # )
    # download(
    #     locations_file="data/labeled/climbing_locations.csv",
    #     adapter=Sentinel2Adapter(output_folder="data/labeled/sentinel2"),
    #     parallel=True,
    #     limit=None,
    # )
    # download(
    #     locations_file="data/labeled/climbing_locations.csv",
    #     adapter=LithologyAdapter(
    #         output_folder="data/labeled/lithology", rate_limiter_ms=500
    #     ),
    #     limit=None,
    #     parallel=True,
    # )

    download(
        locations_file="data/unlabeled/unlabeled_locations.csv",
        adapter=DEMAdapter(output_folder="data/unlabeled/dem"),
        parallel=True,
        limit=None,
    )
    download(
        locations_file="data/unlabeled/unlabeled_locations.csv",
        adapter=Sentinel2Adapter(output_folder="data/unlabeled/sentinel2"),
        parallel=True,
        limit=None,
    )
    download(
        locations_file="data/unlabeled/unlabeled_locations.csv",
        adapter=LithologyAdapter(
            output_folder="data/unlabeled/lithology", rate_limiter_ms=None
        ),
        limit=None,
        parallel=True,
    )

    pass
