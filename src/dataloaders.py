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
        return len(self.set_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / self.set_labels.iloc[idx, 0]
        image = skimage.io.imread(img_name)
        labels = self.set_labels.iloc[idx, 1:]
        labels = self.set_labels.iloc[idx, 1:]
        sample = {"image": image, "labels": labels.values.astype(np.int8)}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

# test if working
if __name__ == "__main__":
    data_path = Path("/home/michal/personal/programming/set-solve/data") 
    set_dataset = SetDataset(csv_file=(data_path / 'labels_final.csv'), root_dir=(data_path / "out"))
    dataloader = DataLoader(set_dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(batch["image"].shape)
    print(batch["labels"].shape)
    print(batch['labels'])