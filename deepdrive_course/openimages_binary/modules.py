import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.functional.classification import accuracy, precision, recall

from deepdrive_course.utils import get_optimizer, get_scheduler


class LitBinaryClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()

        self.config = config
        self.model = model

        if config["pos_class_weight"]:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config["pos_class_weight"]]))
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x).squeeze()
        probas = F.sigmoid(logits).squeeze()
        y_pred = (probas > 0.5).int()

        loss = self.loss_fn(logits, y.float())
        _accuracy = accuracy(y_pred, y, "binary")
        _balanced_accuracy = accuracy(y_pred, y, "multiclass", num_classes=2, average="macro")
        _precision = precision(y_pred, y, "binary")
        _recall = recall(y_pred, y, "binary")

        metrics = {
            "train_loss": loss,
            "train_accuracy": _accuracy,
            "train_balanced_accuracy": _balanced_accuracy,
            "train_precision": _precision,
            "train_recall": _recall,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def on_validation_epoch_start(self):
        self.y_true = []
        self.y_pred = []

    def on_validation_epoch_end(self):
        _balanced_accuracy = balanced_accuracy_score(self.y_true, self.y_pred)

        logs = {
            "val_balanced_accuracy_sklearn": _balanced_accuracy,
        }
        self.log_dict(logs, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.y_true.extend(y.detach().cpu().tolist())

        logits = self(x).squeeze()
        probas = F.sigmoid(logits).squeeze()
        y_pred = (probas > 0.5).int()

        loss = self.loss_fn(logits, y.float())
        _accuracy = accuracy(y_pred, y, "binary")
        _balanced_accuracy = accuracy(y_pred, y, "multiclass", num_classes=2, average="macro")
        _precision = precision(y_pred, y, "binary")
        _recall = recall(y_pred, y, "binary")

        self.y_pred.extend(y_pred.detach().cpu().tolist())

        metrics = {
            "val_loss": loss,
            "val_accuracy": _accuracy,
            "val_balanced_accuracy": _balanced_accuracy,
            "val_precision": _precision,
            "val_recall": _recall,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.config["optimizer"], self.parameters(), lr=self.config["lr"], **self.config["optimizer_kwargs"]
        )
        scheduler = get_scheduler(
            self.config["scheduler"], optimizer, self.config["scheduler_interval"], **self.config["scheduler_kwargs"]
        )

        if scheduler is None:
            return optimizer
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
