import torch
import torch.nn as nn
import torch.nn.functional as F


class model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=1)
        self.fc1 = nn.Linear(32 * 1, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)