from residualblocks import ResidualBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvenBetterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = ResidualBlock(3, 32)
        self.l2 = ResidualBlock(32, 64, stride=2)
        self.l3 = ResidualBlock(64, 128, stride=2)
        self.l4 = ResidualBlock(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 30)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
