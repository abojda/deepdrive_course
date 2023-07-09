This repository contains homework projects developed as a part of [Deepdrive image classification course](https://deepdrive.pl/klasyfikacja/).

# List of the projects
Click on a chapter number to go to the notebooks and results for a given chapter.

| Chapter            | Task                                                                                                  | Main libraries                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [02](notebooks/02) | Perform basic **data visualization** of MNIST-like dataset                                            | numpy                                          |
| [03](notebooks/03) | Train very **simple CNN model** for classification of MNIST-like dataset                              | PyTorch                                        |
| [04](notebooks/04) | Apply various **regularization techniques** to improve model from the previous chapter                | PyTorch Ligthning (PL), Weights & Biases (W&B) |
| [05](notebooks/05) | **Training from scratch vs Transfer Learning** for satellite image classification on RESISC45 dataset | PyTorch Image Models (TIMM), PL, W&B           |
| [06](notebooks/06) | Improve models from previous chapter using **data augmentation**                                      | Albumentations, PL, TIMM, W&B                  |
| [07](notebooks/07) | Run **hyperparamter optimization** (e.g. Optuna) on models from the two previous chapters             | Optuna, PL, TIMM, W&B                          |
| [08](notebooks/08) | **Interpretability** analysis (e.g. occlusion sensitivity and GradCAM) for models from chapters 05-07 | Captum, PL, TIMM, W&B                          |
| [09](notebooks/09) | Run **Self-Supervised Learning** (SSL) on unlabeled dataset as a pretraining for supervised model     | Lightly, PL, TIMM, W&B                         |
| [10](notebooks/10) | **Binary classification with imbalanced dataset** (incorporating weighted loss and balanced accuracy) | PL, TIMM, W&B, FiftyOne                        |
| [11](notebooks/11) | **Model optimization** (e.g. pruning or quantization)                                                 | Intel Neural Compressor, PyTorch               |
| [12](notebooks/12) | **Demo deployment** of one of the models developed in the previous chapters                           | Gradio, PyTorch                                |


# `deepdrive_course` library
The part of the code was put into the [`deepdrive_course`](deepdrive_course) library to reuse the code and make the notebooks more readable.

## Running notebooks in Google Colab
All notebooks using `deepdrive_course` library have the following snippet in the beginning.
```python
import sys
in_colab = "google.colab" in sys.modules

if in_colab:
  !git clone https://github.com/abojda/deepdrive_course.git dd_course
  !pip install dd_course/ -q
```
**If notebook is run from the Google Colab, this reposity is automatically cloned and library is installed.**

## Running notebooks locally
To run notebooks locally, `deepdrive_course` must be installed by hand.

#### 1. Clone this repository
```bash
git clone https://github.com/abojda/deepdrive_course.git`
```

#### 2. Install library
```bash
pip install -e deepdrive_course/ # Dev install
```
or
```bash
pip install deepdrive_course/    # Regular install
```

It is recommended to use virtual environment (venv, conda, ...) to avoid dependency conflicts with other projects.

