import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deepdrive_course.resisc45.datasets import RESISC45, RESISC45Albumentations
from deepdrive_course.utils import stratified_train_test_split


class RESISC45DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        train_transform=None,
        test_transform=None,
        target_transform=None,
        albumentations=False,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.RESISC45 = RESISC45Albumentations if albumentations else RESISC45

    def prepare_data(self):
        self.RESISC45(root=self.root, download=True)

    def setup(self, stage):
        full_ds = self.RESISC45(
            root=self.root,
            download=False,
            transform=self.train_transform,
            target_transform=self.target_transform,
        )

        self.train_ds, self.val_ds, _, _ = stratified_train_test_split(
            dataset=full_ds,
            train_size=0.8,
            train_transform=self.train_transform,
            test_trainsform=self.test_transform,
        )

        if stage == "test":
            self.test_ds = self.val_ds

        if stage == "predict":
            self.predict_ds = self.val_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=True)
