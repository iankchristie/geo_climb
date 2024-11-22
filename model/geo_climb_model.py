import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import pdb
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


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

        self.train_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_auroc = AUROC(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")

    def compute_and_log_metrics(self, outputs, labels, stage):
        accuracy = getattr(self, f"{stage}_accuracy")(outputs, labels)
        precision = getattr(self, f"{stage}_precision")(outputs, labels)
        recall = getattr(self, f"{stage}_recall")(outputs, labels)
        f1 = getattr(self, f"{stage}_f1")(outputs, labels)
        auroc = getattr(self, f"{stage}_auroc")(outputs, labels)

        self.log(
            f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(f"{stage}_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)  # Flatten the output
        loss = self.criterion(outputs, labels.float())
        self.log("train_loss", loss)
        self.compute_and_log_metrics(outputs, labels, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        self.log("val_loss", loss)

        self.compute_and_log_metrics(outputs, labels, stage="val")

        return loss

    def test_step(self, batch, batch_idx):
        data, labels, _, _ = batch
        outputs = self(data).squeeze(1)
        loss = self.criterion(outputs, labels.float())
        self.log("test_loss", loss)

        self.compute_and_log_metrics(outputs, labels, stage="test")

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
