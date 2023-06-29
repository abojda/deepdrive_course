import os
import urllib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


# Version with images saved into .png files and loaded one-by-one
# - Long first-initialization, because of .npy => .png files transformation
class QuickdrawDataset(Dataset):
    def __init__(
        self,
        root: str,
        classes: list[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.target_transform = target_transform

        self.file_paths = []
        self.targets = []

        self._download()
        self._setup()

    def _download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for class_name in self.classes:
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
            url = url.replace(" ", "%20")
            file_path = Path(self.root, f"{class_name}.npy")

            if not os.path.isfile(file_path):
                print(url, "==>", file_path)
                urllib.request.urlretrieve(url, file_path)

    def _setup(self):
        for class_name in self.classes:
            file_path = Path(self.root, f"{class_name}.npy")
            data = np.load(file_path, allow_pickle=True)
            data = rearrange(data, "b (h w) -> b h w", h=28)

            for i, img_data in enumerate(data):
                img_file_path = Path(self.root, f"{class_name}_{i}.png")

                if not os.path.exists(img_file_path):
                    pil_img = Image.fromarray(img_data)
                    pil_img.save(img_file_path)

                self.file_paths.append(img_file_path)
                self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


# Version with images loaded into memory
class QuickdrawDatasetInMemory(QuickdrawDataset):
    def _setup(self):
        for class_name in self.classes:
            file_path = Path(self.root, f"{class_name}.npy")
            data = np.load(file_path, allow_pickle=True)
            data = rearrange(data, "b (h w) -> b h w", h=28)

            for img_data in data:
                self.images.append(Image.fromarray(img_data))
                self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
