import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size, image_size=(28, 28), n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_Dropout(nn.Module):
    def __init__(self, ratios, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Dropout(ratios[0]),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Dropout(ratios[1]),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_MaxPool1(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_MaxPool2(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_MaxPool3(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(11 * 11 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_BatchNorm1(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_BatchNorm2(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_BatchNormFull(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_BatchNorm_Dropout(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_MaxPool2_Dropout(nn.Module):
    def __init__(self, ratio, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratio),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNN_Dropout_MaxPool2(nn.Module):
    def __init__(self, ratio, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Dropout(ratio),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNNBig(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNNBig_MaxPool2_Dropout(nn.Module):
    def __init__(self, ratio, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratio),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNNBig_BatchNorm_MaxPool2_Dropout(nn.Module):
    def __init__(self, ratio, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratio),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 10 * 256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNNBig_MaxPool2_DoubleDropout(nn.Module):
    def __init__(self, ratios, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratios[0]),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(ratios[1]),
            nn.Linear(10 * 10 * 256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class CNNBig_DoubleMaxPool_DoubleDropout(nn.Module):
    def __init__(self, ratios, n_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratios[0]),
            nn.Conv2d(64, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(ratios[1]),
            nn.Flatten(),
            nn.Linear(5 * 5 * 256, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
