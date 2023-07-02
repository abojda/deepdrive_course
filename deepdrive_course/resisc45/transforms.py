import cv2
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(transform_name):
    if transform_name.startswith("albumentations_"):
        return get_albumentations_transform(transform_name)
    elif transform_name.startswith("torchvision_"):
        return get_torchvision_transform(transform_name)
    else:
        raise ValueError(transform_name)


def get_albumentations_transform(transform_name):
    if transform_name == "albumentations_imagenet_norm":
        return A.Compose(
            [
                A.SmallestMaxSize(235, interpolation=cv2.INTER_CUBIC),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    elif transform_name == "albumentations_basic_aug":
        return A.Compose(
            [
                A.SmallestMaxSize(235, interpolation=cv2.INTER_CUBIC),
                A.CenterCrop(224, 224),
                A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomFog(p=0.3),
                        A.Rotate(),
                        A.RandomBrightnessContrast(p=0.3),
                    ]
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(transform_name)


def get_torchvision_transform(transform_name):
    if transform_name == "torchvision_imagenet_norm":
        return T.Compose(
            [
                T.Resize(size=235, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            ]
        )

    elif transform_name == "torchvision_randaugment":
        return T.Compose(
            [
                T.Resize(size=235, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=(224, 224)),
                T.RandAugment(num_ops=3),
                T.ToTensor(),
                T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            ]
        )
    else:
        raise ValueError(transform_name)
