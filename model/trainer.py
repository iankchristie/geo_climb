import pytorch_lightning as pl
from data_module import GeoClimbDataModule
from lightning_module import GeoClimbModel
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir=".", name="lightning_logs", version="sentinel_lithology_1"
)

data_module = GeoClimbDataModule(batch_size=32, data_types=["sentinel", "lithology"])
embedding_size = data_module.get_embedding_size()
model = GeoClimbModel(embedding_size)
trainer = pl.Trainer(max_epochs=50, logger=logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
