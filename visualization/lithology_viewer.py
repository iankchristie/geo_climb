import sys
import os

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.file_utils import *
import json


def view_lithology(
    data_dir: str,
    latitude: float = 40.0150,
    longitude: float = -105.2705,
):
    output_json = encode_file(latitude, longitude, "lit", data_dir, file_type="json")
    with open(output_json, "r") as json_file:
        data = json.load(json_file)
        print(json.dumps(data, indent=4))


if __name__ == "__main__":
    view_lithology(
        data_dir="data/labeled/lithology", latitude=40.0150, longitude=-105.2705
    )
