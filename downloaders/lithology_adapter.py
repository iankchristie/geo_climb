import requests
import os
from filelock import FileLock
import json


class LithologyAdapter:
    def __init__(
        self,
        output_folder: str = "data/lithology",
    ) -> None:
        self.url = "https://macrostrat.org/api/v2/mobile/map_query_v2"
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.output_folder = output_folder
        self.lock_file = os.path.join(self.output_folder, "process.lock")

    # Defaulting to Boulder to make testing easier
    def download(self, latitude: float = 40.0150, longitude: float = -105.2705):
        # z is zoom, testing on the map shows that 12 seems sufficient.
        params = {"lng": -105.2705, "lat": 40.0150, "z": 12}
        response = requests.get(self.url, headers=self.headers, params=params)

        # Create a base filename based on latitude and longitude
        filename_base = f"lit_{latitude}_{longitude}"
        output_json = os.path.join(self.output_folder, f"{filename_base}.json")

        if response.status_code == 200:
            with FileLock(self.lock_file, timeout=10):
                try:
                    # Save the response JSON to a file
                    with open(output_json, "w") as json_file:
                        json.dump(response.json(), json_file, indent=4)
                    print(f"JSON data saved to {output_json}")
                except Exception as e:
                    print(f"Error writing JSON data to file: {e}")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")


if __name__ == "__main__":
    lithology_downloader = LithologyAdapter()
    lithology_downloader.download(latitude=40.0150, longitude=-105.2705)
