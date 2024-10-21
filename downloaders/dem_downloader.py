import pandas as pd
from dem_adapter import *
from file_utils import *
import concurrent.futures
import time


def get_lat_lons(
    data_dir: str,
    limit: int | None = 10,
    csv_file_path="./data/scratch/lat_lons.csv",
) -> list[tuple[float, float]]:
    df = pd.read_csv(csv_file_path)
    lat_lon_pairs = df[["Latitude", "Longitude"]].dropna()
    lat_lon_list = list(lat_lon_pairs.itertuples(index=False, name=None))
    downloaded = get_lat_lons_from_directory(data_dir)
    to_download = [lat_lon for lat_lon in lat_lon_list if lat_lon not in downloaded]
    if limit:
        return to_download[:limit]

    return to_download


def dem_download(
    data_dir: str = "data/dem",
    limit: int | None = 10,
):
    lat_lons = get_lat_lons(data_dir, limit)
    dem_downloader = DEMAdapter(output_folder=data_dir)

    start_time = time.time()
    for lat, lon in lat_lons:
        dem_downloader.download(latitude=lat, longitude=lon)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Finished downloading {len(lat_lons)} DEM files in {elapsed_time:.2f} seconds."
    )


def dem_download_parallel(data_dir: str = "data/dem_parallel", limit: int | None = 10):
    lat_lons = get_lat_lons(data_dir, limit)
    dem_downloader = DEMAdapter(output_folder=data_dir)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all the download tasks to run concurrently
        futures = [
            executor.submit(
                dem_downloader.download,
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
        f"Finished downloading {len(lat_lons)} DEM files in {elapsed_time:.2f} seconds."
    )


if __name__ == "__main__":
    dem_download(limit=100)
    dem_download_parallel(limit=30)
