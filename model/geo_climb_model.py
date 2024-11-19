import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import pdb


class GeoClimbModel(pl.LightningModule):
    def __init__(self, embedding_size, learning_rate=1e-3):
        super(GeoClimbModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)  # Flatten the output
        loss = self.criterion(outputs, labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
