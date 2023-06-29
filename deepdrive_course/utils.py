from copy import copy
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics.functional.classification import confusion_matrix


def stratified_train_test_split(
    dataset, train_size, train_transform=None, test_transform=None, random_state=42
):
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

def get_scheduler(name, **kwargs):
    if name == None:
        return None

    scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    return scheduler_cls(**kwargs)


def plot_to_pil_image(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    return Image.open(buf)


def confusion_matrix_image(
    preds, target, classes, current_epoch, task="multiclass", normalize="true"
):
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
    ax.set(
        xlabel="Predicted label", ylabel="True label", title=f"Epoch {current_epoch}"
    )

    return plot_to_pil_image(fig)
