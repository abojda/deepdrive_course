## Experimenting on 10% of the dataset
First set of experiments was run on 10% of the dataset to speed up the training. Results of these experiments are available as an [interactive Weights & Biases report](https://api.wandb.ai/links/alebojd/qguswr45) or as a [rendered PDF](results_10perc.pdf).

The most promising were models with max pooling and dropout (ratio=0.3) after second CNN layer.

## Experimenting on the full dataset
Models with aforementioned regularization techniques were trained on the full dataset. Results of these experiments are also available as an [interactive Weights & Biases report](https://api.wandb.ai/links/alebojd/tmxrtnjz) or as a [rendered PDF](results_full.pdf). 

**Using regularization techniques, we've managed to:**
- **Reduce the model size by ~4 times, while maintaining efficiency (cnn-maxpool2-dropout_0.3)**
- **Improve validation accuracy by ~2.5 percentage points with model of similar size as the baseline model (cnnmed-maxpool2-dropout_0.3)**
- It looks like there is a potential for further improvements with longer training

`cnn-maxpool2-dropout_0.3` and `cnnmed-maxpool2-dropout_0.3` have different number of convolutional filters in each of the layers.

| Model                         | Number of parameters | Model architecture      |
| ----------------------------- | :------------------- | ----------------------- |
| `cnn-baseline`                | ~160k parameters     | [CNN](https://github.com/abojda/deepdrive_course/blob/main/deepdrive_course/quickdraw/models.py#L26) |
| `cnn-maxpool2-dropout_0.3`    | ~38k parameters      | [CNN_MaxPool2_Dropout](https://github.com/abojda/deepdrive_course/blob/main/deepdrive_course/quickdraw/models.py#L217) |
| `cnnmed-maxpool2-dropout_0.3` | ~167k parameters     | [CNNMed_MaxPool2_Dropout](https://github.com/abojda/deepdrive_course/blob/main/deepdrive_course/quickdraw/models.py#L261) |
