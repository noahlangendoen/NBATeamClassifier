# This is a PyTorch implementation of a Convolutional Neural Network, 
# and is the third model constructed. This model is more complex than previous models,
# and imports residual blocks from the residualblocks.py file to create residual connections.

from models.residualblocks import ResidualBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvenBetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional/Batch Normalization layers with residual connections
        self.l1 = ResidualBlock(3, 32)
        self.l2 = ResidualBlock(32, 64, stride=2)
        self.l3 = ResidualBlock(64, 128, stride=2)
        self.l4 = ResidualBlock(128, 256, stride=2)
        self.l5 = ResidualBlock(256, 512, stride=2)

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 30)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

