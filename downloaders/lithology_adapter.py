import requests
import os
from filelock import FileLock
import json
from adapter import *


class LithologyAdapter(SafeAdapter):
    def __init__(
        self,
        output_folder: str,
    ) -> None:
        self.url = "https://macrostrat.org/api/v2/mobile/map_query_v2"
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        super().__init__(output_folder)

    def download(self, latitude: float, longitude: float):
        # z is zoom, testing on the map shows that 12 seems sufficient.
        params = {"lng": -105.2705, "lat": 40.0150, "z": 12}
        response = requests.get(self.url, headers=self.headers, params=params)

        # Create a base filename based on latitude and longitude
        filename_base = f"lit_{latitude}_{longitude}"
        output_json = os.path.join(self.output_folder, f"{filename_base}.json")

        if response.status_code == 200:
            # This is actually probably OK without the locking, but doing just to be safe.
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
    lithology_downloader = LithologyAdapter(output_folder="data/labeled/lithology")
    lithology_downloader.download(latitude=40.0150, longitude=-105.2705)
