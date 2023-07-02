import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import wandb
from deepdrive_course.utils import confusion_matrix_image, get_optimizer, get_scheduler
from torchmetrics.functional.classification import accuracy, f1_score


class QuickdrawLit(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        loss = F.nll_loss(out, y)
        acc = accuracy(out, y, "multiclass", num_classes=len(self.config["classes"]))
        score = f1_score(out, y, "multiclass", num_classes=len(self.config["classes"]))

        logs = {"train_loss": loss, "train_acc": acc, "train_f1_score": score}
        self.log_dict(logs, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        loss = F.nll_loss(out, y)
        acc = accuracy(out, y, "multiclass", num_classes=len(self.config["classes"]))
        score = f1_score(out, y, "multiclass", num_classes=len(self.config["classes"]))

        logs = {"val_loss": loss, "val_acc": acc, "val_f1_score": score}
        self.log_dict(logs, on_step=False, on_epoch=True, logger=True)

        self.probas.append(out.detach().cpu())
        self.targets.append(y.detach().cpu())

        return loss

    def on_validation_epoch_start(self):
        self.probas = []
        self.targets = []

    def on_validation_epoch_end(self):
        cm_img = confusion_matrix_image(
            preds=torch.cat(self.probas),
            target=torch.cat(self.targets),
            classes=self.config["classes"],
            current_epoch=self.current_epoch,
            task="multiclass",
        )

        self.logger.experiment.log(
            {
                "confusion_matrix": wandb.Image(cm_img),
            }
        )

        self.probas.clear()
        self.targets.clear()

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
