from file_utils import *
import json


def view_lithology(
    latitude: float = 40.0150,
    longitude: float = -105.2705,
    data_dir: str = "data/lithology",
):
    output_json = encode_file(latitude, longitude, "lit", data_dir, file_type="json")
    with open(output_json, "r") as json_file:
        data = json.load(json_file)
        print(json.dumps(data, indent=4))


if __name__ == "__main__":
    view_lithology(latitude=40.0150, longitude=-105.2705)
