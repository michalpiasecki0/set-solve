import os
import sys
from pathlib import Path

import torch
from torchvision.transforms import transforms

COLOR_MAPPING = {"Red": 0, "Green": 1, "Purple": 2}
FILLMENT_MAPPING = {"Empty": 0, "Half": 1, "Full": 2}
SHAPE_MAPPING = {"Diamond": 0, "Oval": 1, "Squiggle": 2}
COUNT_MAPPING = {1: 0, 2: 1, 3: 2}

DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data"


RUN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((120, 120)),
        transforms.ConvertImageDtype(torch.float32),
    ]
)
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((120, 120)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)
