#!/usr/bin/env python3

import gradio as gr
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor

# Classes
class_names = [
    "banana",
    "baseball bat",
    "carrot",
    "clarinet",
    "crayon",
    "pencil",
    "boomerang",
    "hockey stick",
    "fork",
    "knife",
]

# Model definition
class CNNBig_MaxPool2_Dropout(nn.Module):
    def __init__(self, ratio):
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
            nn.Linear(10 * 10 * 256, len(class_names)),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBig_MaxPool2_Dropout(0.4)
model.load_state_dict(torch.load("cnnbig-maxpool2-dropout_0.4__5_epochs.pt"))
model.to(device)
model.eval()

# Define image transform
transform = ToTensor()


def predict(img):
    with torch.inference_mode():
        img = transform(img).unsqueeze(0)
        log_probas = model(img)
        probas = torch.exp(log_probas).detach().cpu().squeeze().tolist()

    prediction = {label: prob for label, prob in zip(class_names, probas)}
    return prediction


demo = gr.Interface(
    fn=predict, inputs=gr.Image(type="pil", source="canvas", image_mode="L", shape=(28, 28)), outputs=["label"]
)
demo.launch()
