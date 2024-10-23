import requests
import os
import json
from safe_adapter import *


class LithologyAdapter(SafeAdapter):
    def __init__(
        self, output_folder: str, rate_limiter_ms: float | None = None
    ) -> None:
        self.url = "https://macrostrat.org/api/v2/mobile/map_query_v2"
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        super().__init__(output_folder=output_folder, rate_limiter_ms=rate_limiter_ms)

    def pull_data(self, latitude: float, longitude: float) -> requests.Response | None:
        # z is zoom, testing on the map shows that 12 seems sufficient.
        params = {"lng": longitude, "lat": latitude, "z": 12}
        return requests.get(self.url, headers=self.headers, params=params)

    def write_data(
        self, response: requests.Response, latitude: float, longitude: float
    ):
        filename_base = f"lit_{latitude}_{longitude}"
        output_json = os.path.join(self.output_folder, f"{filename_base}.json")
        try:
            # Save the response JSON to a file
            with open(output_json, "w") as json_file:
                json.dump(response.json(), json_file, indent=4)
            print(f"JSON data saved to {output_json}")
        except Exception as e:
            print(f"Error writing JSON data to file: {e}")


if __name__ == "__main__":
    lithology_downloader = LithologyAdapter(output_folder="data/labeled/lithology")
    lithology_downloader.download(latitude=40.0150, longitude=-105.2705)
