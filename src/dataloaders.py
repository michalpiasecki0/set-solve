import io
import os
from pathlib import Path

import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SetDataset(Dataset):
    
    """
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.set_labels = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / self.set_labels.iloc[idx, 0]
        image = skimage.io.imread(img_name)
        labels = self.set_labels.iloc[idx, 1:]
        sample = {"image": image, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
