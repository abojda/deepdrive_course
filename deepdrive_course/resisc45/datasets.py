import os
import shutil

import cv2
from PIL import Image
from torch.utils.data import Dataset

import gdown
import patoolib


class RESISC45(Dataset):
    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        self.filepaths = []
        self.targets = []

        for idx, _class in enumerate(self.classes):
            directory = os.path.join(self.root, _class)
            files = os.listdir(directory)
            filepaths = [os.path.join(self.root, _class, file) for file in files]
            self.filepaths.extend(filepaths)
            self.targets.extend([idx] * len(filepaths))

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

    def _download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        url = "https://drive.google.com/uc?id=1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
        rar_filepath = os.path.join(self.root, "resisc45.rar")

        if not os.path.isfile(rar_filepath):
            gdown.download(url, rar_filepath)

        extracted_dir = os.path.join(self.root, "NWPU-RESISC45")

        # Ensure that extraction directory does not exsist
        if os.path.isdir(extracted_dir):
            shutil.rmtree(extracted_dir)

        patoolib.extract_archive(rar_filepath, outdir=self.root)

        for class_name in os.listdir(extracted_dir):
            source_class_dir = os.path.join(extracted_dir, class_name)
            destination_class_dir = os.path.join(self.root, class_name)

            # Move only directories that don't exist
            if not os.path.isdir(destination_class_dir):
                shutil.move(source_class_dir, self.root)
            else:
                shutil.rmtree(source_class_dir)

        os.rmdir(extracted_dir)


class RESISC45Albumentations(RESISC45):
    def __getitem__(self, idx):
        img = cv2.imread(self.filepaths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.targets[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.target_transform:
            target = self.target_transform(target)

        return img, target
