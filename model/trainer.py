import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.geo_climb_data_module import GeoClimbDataModule
from model.evaluator import evaluate_model
from model.geo_climb_model import GeoClimbModel
from torchgeo import trainers
from pytorch_lightning.loggers import WandbLogger

EMBEDDING_DIRECTORIES = ["sentinel_mosaiks", "dem_v2", "lithology_v2"]
name = "__".join(EMBEDDING_DIRECTORIES)

wandb_logger = WandbLogger(project="geo-climb", name=name)

data_module = GeoClimbDataModule(batch_size=32, name_encoding=name)
embedding_size = data_module.get_embedding_size()
model = GeoClimbModel(embedding_size)
trainer = pl.Trainer(max_epochs=100, logger=wandb_logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

evaluate_model(model, data_module.test_dataset)
