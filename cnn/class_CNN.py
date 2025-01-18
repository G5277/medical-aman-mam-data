import torch
import torch.nn as nn
import torch.nn.functional as F

class model_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,2, padding=1)
        self.conv2 = nn.Conv1d(16,32,2, padding=1)
        self.fc1 = nn.Linear(160,64)
        self.fc2 = nn.Linear(64,3)

    def convs(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    def forward(self,x):
        x = self.convs(x)
        # print("Input :",x.shape)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)