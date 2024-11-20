import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.geo_climb_data_module import GeoClimbDataModule
from model.evaluator import evaluate_model
from model.geo_climb_model import GeoClimbModel


logger = TensorBoardLogger(
    save_dir=".", name="lightning_logs", version="sentinel_mosaiks_dem_lit_1"
)

data_module = GeoClimbDataModule(
    batch_size=32, data_types=["sentinel", "dem", "lithology"]
)
embedding_size = data_module.get_embedding_size()
model = GeoClimbModel(embedding_size)
trainer = pl.Trainer(max_epochs=100, logger=logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

evaluate_model(model, data_module.test_dataset)
