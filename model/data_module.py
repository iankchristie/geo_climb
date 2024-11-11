import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
from data_set import GeoClimbDataset


class GeoClimbDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Load the datasets. This is called before training, validation, and testing.
        The `stage` parameter can be 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = GeoClimbDataset(split="training")
            self.val_dataset = GeoClimbDataset(split="validation")

        if stage == "test" or stage is None:
            self.test_dataset = GeoClimbDataset(split="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
