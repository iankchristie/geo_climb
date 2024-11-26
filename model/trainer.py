import os
import sys
import pytorch_lightning as pl
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.geo_climb_data_module import GeoClimbDataModule
from model.evaluator import evaluate_model
from model.geo_climb_model import GeoClimbModel
from pytorch_lightning.loggers import WandbLogger

with open("model/config.yml", "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters
project_name = config["project_name"]
embedding_directories = config["embedding_directories"]
hyperparameters = config["hyperparameters"]

name = "__".join(embedding_directories)
batch_size = hyperparameters["batch_size"]
max_epochs = hyperparameters["max_epochs"]
learning_rate = hyperparameters["learning_rate"]

wandb_logger = WandbLogger(project="geo-climb", name=name)
wandb_logger.experiment.config.update(
    {
        "embedding_directories": embedding_directories,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
    }
)

if torch.cuda.is_available():
    accelerator = "cuda"
elif torch.backends.mps.is_available():
    accelerator = "mps"
else:
    accelerator = "cpu"

data_module = GeoClimbDataModule(batch_size=batch_size, name_encoding=name)
embedding_size = data_module.get_embedding_size()
model = GeoClimbModel(embedding_size, learning_rate=learning_rate)
trainer = pl.Trainer(accelerator=accelerator,max_epochs=max_epochs, logger=wandb_logger, fast_dev_run=False)
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)

evaluate_model(model, data_module.test_dataset)
