from abc import ABC, abstractmethod
import os
from filelock import FileLock
from requests import Response
import time
import random


class SafeAdapter(ABC):
    def __init__(self, output_folder: str, rate_limiter_ms: float | None = None):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.lock_file = os.path.join(self.output_folder, "process.lock")
        self.rate_limiter_ms = rate_limiter_ms

    def download(self, latitude, longitude):
        self.rate_limit()
        response = self.pull_data(latitude, longitude)
        if response and response.status_code == 200:
            with FileLock(self.lock_file, timeout=20):
                self.write_data(response, latitude, longitude)
        else:
            print(f"Failed to download data for coordinates ({latitude}, {longitude}).")

    def rate_limit(self):
        if self.rate_limiter_ms:
            lower_bound = self.rate_limiter_ms - self.rate_limiter_ms / 2
            upper_bound = self.rate_limiter_ms + self.rate_limiter_ms / 2
            sleep_time = random.uniform(lower_bound, upper_bound) / 1000
            time.sleep(sleep_time)

    @abstractmethod
    def pull_data(self, latitude: float, longitude: float) -> Response | None:
        pass

    @abstractmethod
    def write_data(self, response: Response, latitude: float, longitude: float):
        pass
