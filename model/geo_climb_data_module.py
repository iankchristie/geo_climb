import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
from model.geo_climb_data_set import GeoClimbDataset


class GeoClimbDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_types=["lithology", "sentinel"]):
        super().__init__()
        self.batch_size = batch_size
        self.data_types = data_types
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Load the datasets. This is called before training, validation, and testing.
        The `stage` parameter can be 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = GeoClimbDataset(
                split="training", data_types=self.data_types
            )
            self.val_dataset = GeoClimbDataset(
                split="validation", data_types=self.data_types
            )

        if stage == "test" or stage is None:
            self.test_dataset = GeoClimbDataset(
                split="test", data_types=self.data_types
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_embedding_size(self) -> int:
        # Creating this is not ideal, but when we need to create the model we haven't actually "setup" the datamodule yet.
        # So we can create a dummy dataset with the correct data_types.
        return GeoClimbDataset(
            split="test", data_types=self.data_types
        ).get_embedding_size()