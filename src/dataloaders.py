from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


class SetDataset(Dataset):
    """
    Dataset for segmented imgs with set cards and corresponding labels
    """

    def __init__(self, csv_file: Path, root_dir: Path, transform: Callable = None):
        """
        Args:
            csv_file (str): path to csv with ground truth labels
            root_dir (str):
            transform ([type], optional): [description]. Defaults to None.
        """
        self.set_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
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
        # image = np.array(image)

        # sample = {"image": image, "labels": labels.values.astype(np.int8)}

        return image, labels.values.astype(np.int8)

    def split_indices_train_test(self, split_ratio: int) -> Tuple[List]:
        """
        Split dataset into train and test set with given ratio.
        Args:
            split_ratio (int): train / test ratio
        Returns:
            Tuple[List]: [indices_for_train], [indices_for_test]
        """
        dataset_size = len(self)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)  # shuffle indices in place
        split_point = int(np.floor(split_ratio * dataset_size))
        return indices[:split_point], indices[split_point:]


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
    dataset_size = len(set_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_subset = Subset(set_dataset, train_indices)
    val_subset = Subset(set_dataset, val_indices)
    dataloader = DataLoader(set_dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(batch["image"].shape)
    print(batch["labels"].shape)
    print(batch["labels"])
    PIL.Image.fromarray(batch["image"][0].numpy()).save("test.jpg")
