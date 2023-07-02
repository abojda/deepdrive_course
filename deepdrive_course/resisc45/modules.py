import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torchmetrics.functional.classification import accuracy, f1_score

from deepdrive_course.utils import confusion_matrix_image, get_optimizer, get_scheduler


class ResiscLit(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probas = F.log_softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)
        acc = accuracy(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))
        score = f1_score(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))

        logs = {"train_loss": loss, "train_acc": acc, "train_f1_score": score}
        self.log_dict(logs, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probas = F.log_softmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)
        acc = accuracy(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))
        score = f1_score(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))

        logs = {"val_loss": loss, "val_acc": acc, "val_f1_score": score}
        self.log_dict(logs, on_step=False, on_epoch=True, logger=True)

        self.probas.append(torch.exp(log_probas).detach().cpu())
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

    def on_test_epoch_start(self):
        self.test_misclassified = []
        self.test_properly_classified = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probas = F.log_softmax(logits, dim=1)
        probas = torch.exp(log_probas)

        true_labels = y.detach().cpu().numpy()
        pred_labels = torch.argmax(log_probas, dim=1).detach().cpu().numpy()

        misclassified_idx = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true != pred]
        properly_classified_idx = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == pred]

        for idx in misclassified_idx:
            img = x[idx].detach().cpu()
            pred_label = int(pred_labels[idx])
            true_label = int(true_labels[idx])
            prob = probas[idx].detach().cpu()

            self.test_misclassified.append((img, pred_label, true_label, prob))

        for idx in properly_classified_idx:
            img = x[idx].detach().cpu()
            pred_label = int(pred_labels[idx])
            true_label = int(true_labels[idx])
            prob = probas[idx].detach().cpu()

            self.test_properly_classified.append((img, pred_label, true_label, prob))

        loss = F.cross_entropy(logits, y)
        acc = accuracy(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))
        score = f1_score(log_probas, y, "multiclass", num_classes=len(self.config["classes"]))

        metrics = {"test_loss": loss, "test_acc": acc, "test_f1_score": score}
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        return metrics

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
