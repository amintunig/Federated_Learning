import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
import flwr as fl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetCIFAR(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool4(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_weights(self, parameters: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)

def load_cifar10(partition_id: int, num_clients: int, batch_size: int = 32):
    """Load CIFAR-10 dataset and create a DataLoader for a specific client."""
    transform = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./data_cifar10", train=True, download=True, transform=transform)
    testset = CIFAR10("./data_cifar10", train=False, download=True, transform=transform)

    # Partition the training data among clients
    partition_size = len(trainset) // num_clients
    indices = list(range(len(trainset)))
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size
    train_indices = indices[start:end]
    trainloader = DataLoader(
        torch.utils.data.Subset(trainset, train_indices),
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader

def get_cifar10_model():
    """Returns an instance of the CIFAR-10 model."""
    return NetCIFAR().to(DEVICE)