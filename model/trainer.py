import pytorch_lightning as pl
from data_module import GeoClimbDataModule
from lightning_module import GeoClimbModel
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir=".", name="lightning_logs")

data_module = GeoClimbDataModule(batch_size=32)
model = GeoClimbModel()
trainer = pl.Trainer(max_epochs=50, logger=logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
