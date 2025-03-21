from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class SetDataset(Dataset):
    """
    Dataset for segmented imgs with set cards and corresponding labels
    """

    def __init__(self, csv_file: str, root_dir: str, transform: Callable = None):
        """
        Args:
            csv_file ([type]): [description]
            root_dir ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
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
        image = PIL.Image.open(img_name)
        labels = self.set_labels.iloc[idx, 1:]
        labels = self.set_labels.iloc[idx, 1:]

        if self.transform:
            image = self.transform(image)
        image = np.array(image)

        sample = {"image": image, "labels": labels.values.astype(np.int8)}

        return sample


# test if working
if __name__ == "__main__":
    data_path = Path("/home/michal/personal/programming/set-solve/data")
    set_dataset = SetDataset(
        csv_file=(data_path / "labels_final.csv"),
        root_dir=(data_path / "out"),
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Grayscale(),
            ]
        ),
    )
    dataloader = DataLoader(set_dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(batch["image"].shape)
    print(batch["labels"].shape)
    print(batch["labels"])
    PIL.Image.fromarray(batch["image"][0].numpy()).save("test.jpg")
