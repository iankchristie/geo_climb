import os


def get_lat_lons_from_directory(directory: str) -> set[tuple[float, float]]:
    lat_lon_set = set()

    for file_name in os.listdir(directory):
        if file_name.endswith(".tif"):
            try:
                lat_lon = tuple(decode_file(file_name))
                lat_lon_set.add(lat_lon)
            except (ValueError, IndexError):
                print(f"Skipping file with incorrect format: {file_name}")

    return lat_lon_set


def encode_file(latitude: float, longitude: float, data: str) -> str:
    return f"{data}_{latitude}_{longitude}"


def decode_file(file_name: str) -> list[float]:
    file_name = file_name.replace(".tif", "")

    # Split the filename into parts by underscore
    parts = file_name.split("_")

    # Assuming the last two parts are latitude and longitude
    latitude = float(parts[-2])
    longitude = float(parts[-1])

    return [latitude, longitude]
