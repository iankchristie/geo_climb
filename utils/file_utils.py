import os
import pandas as pd


def get_undownloaded_lat_lons(
    download_path: str,
    downloaded_path: str | None = None,
    limit: int | None = 10,
) -> list[tuple[float, float]]:
    df = pd.read_csv(download_path)
    lat_lon_pairs = df[["Latitude", "Longitude"]].dropna()
    lat_lon_list = list(lat_lon_pairs.itertuples(index=False, name=None))
    if not downloaded_path:
        return lat_lon_list

    downloaded = get_lat_lons_from_directory(downloaded_path)
    to_download = [lat_lon for lat_lon in lat_lon_list if lat_lon not in downloaded]
    if limit:
        return to_download[:limit]

    return to_download


def save_lat_lons_to_csv(lat_lons: set[tuple[float, float]], output_file: str):
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(lat_lons, columns=["Latitude", "Longitude"])
    df.to_csv(output_file, index=False)
    print(f"Latitude/Longitude pairs saved to {output_file}")


def get_lat_lons_from_directory(directory: str) -> set[tuple[float, float]]:
    lat_lon_set = set()

    for file_name in os.listdir(directory):
        if file_name.endswith(".tif") or file_name.endswith(".json"):
            try:
                lat_lon = tuple(decode_file(file_name))
                lat_lon_set.add(lat_lon)
            except (ValueError, IndexError):
                print(f"Skipping file with incorrect format: {file_name}")

    return lat_lon_set


def encode_file(
    latitude: float | str,
    longitude: float | str,
    data: str,
    data_dir: str | None = None,
    file_type: str = "tif",
) -> str:
    filename_base = f"{data}_{latitude}_{longitude}.{file_type}"
    if data_dir:
        return os.path.join(data_dir, f"{filename_base}")
    return filename_base


def decode_file(file_name: str) -> list[float]:
    # Remove the file extension by splitting on the last period
    file_name, _ = os.path.splitext(file_name)

    # Split the filena me into parts by underscore
    parts = file_name.split("_")

    # Assuming the last two parts are latitude and longitude
    latitude = float(parts[-2])
    longitude = float(parts[-1])

    return [latitude, longitude]


def delete_file(file_name, directory):
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"File '{file_name}' successfully deleted from {directory}.")
        except Exception as e:
            print(f"Error deleting file '{file_name}': {e}")
    else:
        print(f"File '{file_name}' does not exist in the directory {directory}.")


def os_safe_file_path(file_path):
    if os.name == "nt":  # Check if the OS is Windows
        return file_path.replace("/", "\\")
    else:  # Ensure forward slashes for Unix-based systems (Linux, macOS)
        return file_path.replace("\\", "/")
