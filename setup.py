from setuptools import setup, find_packages

setup(
    name="deepdrive_course",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "wandb",
        "torchsummary",
        "torchmetrics",
        "seaborn",
        "einops",
        "timm",
        "gdown",
        "patool",
        "albumentations",
        "opencv-python",
        "mega.py"
    ],
)
