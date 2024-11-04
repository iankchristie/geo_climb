import os
import sys
import json
import pdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloaders.file_utils import *


def load_and_print_json_files(directory: str):
    result = []
    count_empty_map_data = 0
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as json_file:
                file_data = {}
                try:
                    data = json.load(json_file)
                    lat, lon = decode_file(filename)
                    file_data["Latitude"] = lat
                    file_data["Longitude"] = lon
                    if len(data["success"]["data"]["mapData"]) < 1:
                        count_empty_map_data = count_empty_map_data + 1
                        file_data["lith"] = ""
                        file_data["liths"] = []
                    else:
                        mapData = data["success"]["data"]["mapData"][0]
                        file_data["lith"] = mapData["lith"]
                        lith_data = []
                        for lith in mapData["liths"]:
                            lith_data.append(
                                {
                                    "lith_id": lith.get("lith_id"),
                                    "lith": lith.get("lith"),
                                    "lith_class": lith.get("lith_class"),
                                    "lith_type": lith.get("lith_type"),
                                }
                            )
                        file_data["liths"] = lith_data

                    result.append(file_data)
                except json.JSONDecodeError as e:
                    print(f"Error reading {filename}: {e}")
    print(result[0:10])
    print(count_empty_map_data)


def export_to_csv(data, output_file):
    """
    Export the provided data to a CSV file, flattening the 'liths' array to have one row per lith.

    Parameters:
    - data: List of dictionaries containing latitude, longitude, lith, and liths array.
    - output_file: Path to the output CSV file.
    """
    # Define the fieldnames (columns for the CSV)
    fieldnames = [
        "Latitude",
        "Longitude",
        "lith",
        "lith_id",
        "lith",
        "lith_class",
        "lith_type",
    ]

    # Open the CSV file for writing
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Iterate over each item in the data
        for entry in data:
            # For each entry, go through the 'liths' array
            for lith_entry in entry["liths"]:
                # Write a row for each lith, including the main details
                writer.writerow(
                    {
                        "Latitude": entry.get("Latitude"),
                        "Longitude": entry.get("Longitude"),
                        "lith": entry.get("lith", ""),
                        "lith_id": lith_entry.get("lith_id"),
                        "lith_class": lith_entry.get("lith_class"),
                        "lith_type": lith_entry.get("lith_type"),
                    }
                )

    print(f"Data has been written to {output_file}")


# Example usage:
directory_path = "./data/labeled/lithology"
load_and_print_json_files(directory_path)
