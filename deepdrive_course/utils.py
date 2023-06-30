from copy import copy
from io import BytesIO

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import timm
import torch
from PIL import Image
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics.functional.classification import confusion_matrix

from mega import Mega


def stratified_train_test_split(dataset, train_size, train_transform=None, test_transform=None, random_state=42):
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        range(len(dataset)),
        train_size=train_size,
        stratify=dataset.targets,
        random_state=random_state,
    )

    train_ds = Subset(dataset, train_indices)
    test_ds = Subset(dataset, test_indices)

    if train_transform:
        # Workaround for different train/test transforms when using torch.utils.data.Subset objects
        train_ds.dataset = copy(dataset)
        train_ds.dataset.transform = train_transform

    if test_transform:
        # Workaround for different train/test transforms when using torch.utils.data.Subset objects
        test_ds.dataset = copy(dataset)
        test_ds.dataset.transform = test_transform

    return train_ds, test_ds, train_indices, test_indices


def get_optimizer(name, model_parameters, **kwargs):
    optimizer_cls = getattr(torch.optim, name)
    return optimizer_cls(model_parameters, **kwargs)


def get_scheduler(name, optimizer, interval="epoch", **kwargs):
    if name == None:
        return None

    scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    scheduler = scheduler_cls(optimizer, **kwargs)

    return {
        "scheduler": scheduler,
        "interval": interval,
    }


def plot_to_pil_image(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    return Image.open(buf)


def confusion_matrix_image(preds, target, classes, current_epoch, task="multiclass", normalize="true"):
    cm = confusion_matrix(
        preds=preds,
        target=target,
        task=task,
        num_classes=len(classes),
        normalize=normalize,
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, ax=ax, annot=True)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticklabels(classes, rotation=0, ha="center", va="center")
    ax.tick_params(axis="y", pad=40)
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"Epoch {current_epoch}")

    return plot_to_pil_image(fig)


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def pl_print_model_summary(model, depth=1):
    summary = ModelSummary(model, max_depth=depth)
    print(summary)


def pl_find_max_batch_size(model, datamodule):
    trainer = pl.Trainer(max_epochs=1)
    tuner = pl.tuner.Tuner(trainer)
    max_batch_size = tuner.scale_batch_size(model, mode="power", datamodule=datamodule)
    return max_batch_size


def timm_prepare_params_for_training(timm_model, training_type):
    if training_type == "full" or training_type == "finetuning":
        unfreeze_params(timm_model)
    elif training_type == "transfer_learning":
        freeze_params(timm_model)
        unfreeze_params(timm_model.get_classifier())
    else:
        raise ValueError(training_type)


def timm_get_pretrained_data_transform(timm_model):
    timm_cfg = timm.data.resolve_data_config(timm_model.pretrained_cfg)
    return timm.data.create_transform(**timm_cfg)


def download_from_mega_nz(url):
    m = Mega().login()
    path = m.download_url(url)
    return path
