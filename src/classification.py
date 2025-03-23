import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloaders import SetDataset

class MultiOutputCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(MultiOutputCNN, self).__init__()

        # Feature extraction backbone
        # Input: 3 x 120 x 120
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # -> 32 x 120 x 120
            nn.ReLU(),
            nn.MaxPool2d(4, 4),  # -> 32 x 30 x 30 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),  # -> 64 x 7 x 7
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU())

        # Four independent classification heads (each predicting one of 3 values)
        self.head_color = nn.Linear(256, num_classes)
        self.head_count = nn.Linear(256, num_classes)
        self.head_fill = nn.Linear(256, num_classes)
        self.head_shape = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)

        # predictions
        return self.head_color(x), self.head_count(x), self.head_fill(x), self.head_shape(x)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    epoch_number: int,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epoch_number):
        for i, batch in tqdm(enumerate(dataloader)):
            model.train()
            x, y = batch["image"].to(device), batch["labels"].to(device, dtype=torch.long)
            print(x.dtype, y.dtype)
            #exit()
            y = y.permute((1, 0))
            optimizer.zero_grad()
            out = model(x)
            loss_color = criterion(out[0], y[0])
            loss_count = criterion(out[1], y[1])
            loss_fill = criterion(out[2], y[2])
            loss_shape = criterion(out[3], y[3])
            total_loss = loss_color + loss_count + loss_fill + loss_shape

            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss.item()}")


if __name__ == "__main__":
    model = MultiOutputCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)

    data_path = Path("/home/michal/personal/programming/set-solve/data")
    set_dataset = SetDataset(
        csv_file=(data_path / "labels_final.csv"),
        root_dir=(data_path / "out"),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((120, 120)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
    )
    dataloader = DataLoader(set_dataset, batch_size=15)

    train(model, optimizer, dataloader, 10, device)