import pytorch_lightning as pl
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from deepdrive_course.utils import get_optimizer, get_scheduler


class LitSimCLR(pl.LightningModule):
    def __init__(self, backbone, config):
        super().__init__()
        self.config = config

        self.backbone = backbone
        self.simclr_head = SimCLRProjectionHead(config["input_dim"], config["hidden_dim"], config["output_dim"])
        self.loss = NTXentLoss()

        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, x):
        out = self.backbone(x)
        out = self.simclr_head(out)
        return out

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        loss = self.loss(z0, z1)
        self.log("train_loss_ssl", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        loss = self.loss(z0, z1)
        self.log("val_loss_ssl", loss, on_step=False, on_epoch=True)
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
