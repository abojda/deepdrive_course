from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from copy import copy


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

    return train_ds, test_ds
