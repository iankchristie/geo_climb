import pytorch_lightning as pl
from data_module import GeoClimbDataModule
from model.evaluator import evaluate_model
from lightning_module import GeoClimbModel
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger(
    save_dir=".", name="lightning_logs", version="sen_rfc_gaussian"
)

data_module = GeoClimbDataModule(batch_size=32, data_types=["sentinel"])
embedding_size = data_module.get_embedding_size()
model = GeoClimbModel(embedding_size)
trainer = pl.Trainer(max_epochs=50, logger=logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

evaluate_model(model, data_module.test_dataset)
