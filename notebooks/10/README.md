# Open in Google Colab
`openimages_binary_pl.ipynb`: <a target="_blank" href="https://colab.research.google.com/github/abojda/deepdrive_course/blob/main/notebooks/10/10.openimages_binary_pl.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Results
Two "similar" classes from OpenImages dataset were selected to form imbalanced binary classification problem.

Classes:
- `Sink` - 3408 images (~80.1% of the dataset)
- `Bathtub` - 846 images (~19.9% of the dataset)

## Training with "regular" loss vs with weighted loss
![binary_results](img/binary_results.png)

We can clearly see that training with weighted loss performs much better. First of all, since we have imbalanced data we should monitor balanced accuracy instead of the "regular" accuracy. Model with weighted loss achieved better balanced accuracy (by ~11 percentage points) compared to the model without weighted loss.
