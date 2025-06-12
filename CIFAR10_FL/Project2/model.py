import torch.nn as nn
import torch.nn.functional as F

class MIMICModel(nn.Module):
    def __init__(self, input_dim=100, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
