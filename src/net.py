# This file contains the neural network architecture for the classification task
import torch
import torch.nn as nn


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
        return torch.stack(
            (
                self.head_color(x),
                self.head_count(x),
                self.head_fill(x),
                self.head_shape(x),
            )
        )
