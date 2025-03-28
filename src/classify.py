import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from typing import Tuple

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from dataloaders import SetDataset
from net import MultiOutputCNN
from constants import DATA_PATH

from clearml import Task, Logger


BATCH_SIZE = 16


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_path: Path,
        labels_path: Path,
        transform: transforms.Compose,
        lr: float,
        epoch_number: int,
        train_test_split_ratio: float,
        device: torch.device,
    ):
        self.device = device

        # get model
        self.model = model.to(self.device)

        # get dataloaders
        self.dataset = SetDataset(
            csv_file=labels_path,
            root_dir=dataset_path,
            transform=transform,
        )
        self.train_dataloader, self.test_dataloader = self._get_dataloaders(
            batch_size=BATCH_SIZE, train_test_split_ratio=train_test_split_ratio
        )

        # others
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_number = epoch_number

        # ClearML logger
        self.logger = Logger.current_logger()

    def _get_dataloaders(
        self, batch_size: int, train_test_split_ratio: float
    ) -> Tuple[DataLoader]:
        """
        Return train and test dataloaders from SetDataset
        Returns:
            Tuple[DataLoader]: train_dataloader, test_dataloader
        """
        train_indices, test_indices = self.dataset.split_indices_train_test(
            train_test_split_ratio
        )
        train_subset = Subset(self.dataset, train_indices)
        test_subset = Subset(self.dataset, test_indices)
        train_dataloader = DataLoader(train_subset, batch_size=batch_size)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size)
        return train_dataloader, test_dataloader

    def train(self):
        for epoch in tqdm(range(self.epoch_number)):
            # train single epoch
            total_loss = self.train_epoch()

            # report train loss from epoch
            print(f"Epoch: {epoch}, Train Loss: {total_loss}")
            self.logger.report_scalar(
                "loss", "train", iteration=epoch, value=total_loss
            )

            # report test accuracies per category
            val_loss, val_accuracy = self.evaluate(self.test_dataloader)
            print(f"Epoch: {epoch}, Test Accuracy: {val_accuracy}")
            self._log_all_accuracies(val_accuracy, epoch)
            self.logger.report_scalar("loss", "test", iteration=epoch, value=val_loss)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for img_batch, label_batch in self.train_dataloader:
            img_batch, label_batch = (
                img_batch.to(self.device),
                label_batch.to(self.device, dtype=torch.long),
            )
            label_batch = label_batch.permute((1, 0))  # we need to permute
            self.optimizer.zero_grad()
            out = self.model(img_batch)
            out_color, out_count, out_fill, out_shape = torch.unbind(out, dim=0)
            loss_color = self.criterion(out_color, label_batch[0])
            loss_count = self.criterion(out_count, label_batch[1])
            loss_fill = self.criterion(out_fill, label_batch[2])
            loss_shape = self.criterion(out_shape, label_batch[3])

            loss = loss_color + loss_count + loss_fill + loss_shape

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss

    def evaluate_batch(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate single batch
        Args:
            predictions (torch.Tensor]): category x batch x values
            labels (torch.Tensor): category x batch
        """
        with torch.no_grad():
            n_correct = (torch.argmax(predictions, dim=2) == labels).sum(dim=1)
        return n_correct  # [color_correct, count_correct, fill_correct, shape_correct]

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, torch.Tensor]:
        """
        Calculate validation loss and accuracy for each category to evaluate model
        Args:
            dataloader (DataLoader):

        Returns:
            Tuple[float, torch.Tensor]: Validation loss, Accuracy per category
        """
        self.model.eval()
        total_correct = torch.zeros((4,)).to(self.device)
        total_observations = 0
        total_val_loss = 0
        with torch.no_grad():
            for img_batch, label_batch in dataloader:
                img_batch, label_batch = (
                    img_batch.to(self.device),
                    label_batch.to(self.device, dtype=torch.long),
                )
                label_batch = label_batch.permute((1, 0))
                out = self.model(img_batch)

                # calculate validation loss
                out_color, out_count, out_fill, out_shape = torch.unbind(out, dim=0)
                loss_color = self.criterion(out_color, label_batch[0])
                loss_count = self.criterion(out_count, label_batch[1])
                loss_fill = self.criterion(out_fill, label_batch[2])
                loss_shape = self.criterion(out_shape, label_batch[3])

                loss = loss_color + loss_count + loss_fill + loss_shape
                total_val_loss += loss.item()

                total_correct += self.evaluate_batch(out, label_batch)
                total_observations += img_batch.shape[0]

        return total_val_loss, (total_correct / total_observations)

    def save_model(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def _log_all_accuracies(self, accuracies: torch.Tensor, iteration: int):
        categories = ["color", "count", "fill", "shape"]
        for i, category in enumerate(categories):
            self.logger.report_scalar(
                f"accuracy_{category}", "test", iteration=iteration, value=accuracies[i]
            )


if __name__ == "__main__":
    # task = Task.init(project_name="SetSolve", task_name="Training")

    # ------------ Load dataset ---------------------------
    set_dataset = SetDataset(
        csv_file=(DATA_PATH / "labels_final.csv"),
        root_dir=(DATA_PATH / "out"),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((120, 120)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
    )

    # ----------------- Train -------------------
    model = MultiOutputCNN()
    trainer = Trainer(
        model=model,
        dataset_path=DATA_PATH / "out",
        labels_path=DATA_PATH / "labels_final.csv",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((120, 120)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
        lr=0.001,
        epoch_number=40,
        train_test_split_ratio=0.8,
        device=torch.device("cuda"),
    )
    trainer.train()
    trainer.save_model(DATA_PATH / "set_classifier_simple.pth")
