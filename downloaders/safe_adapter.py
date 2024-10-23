from abc import ABC, abstractmethod
import os
from filelock import FileLock
from requests import Response


class SafeAdapter(ABC):
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.lock_file = os.path.join(self.output_folder, "process.lock")

    def download(self, latitude, longitude):
        response = self.pull_data(latitude, longitude)
        if response and response.status_code == 200:
            with FileLock(self.lock_file, timeout=20):
                self.write_data(response, latitude, longitude)
        else:
            print(f"Failed to download data for coordinates ({latitude}, {longitude}).")

    @abstractmethod
    def pull_data(self, latitude: float, longitude: float) -> Response | None:
        pass

    @abstractmethod
    def write_data(self, response: Response, latitude: float, longitude: float):
        pass
