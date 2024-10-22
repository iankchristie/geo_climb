from abc import ABC, abstractmethod
import os


class SafeAdapter(ABC):
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.lock_file = os.path.join(self.output_folder, "process.lock")

    @abstractmethod
    def download(self, latitude, longitude):
        pass
