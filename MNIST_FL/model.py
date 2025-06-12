import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import flwr as fl  # Import the flwr library

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_weights(self, parameters: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.load_state_dict(state_dict, strict=True)

def load_mnist(partition_id: int, num_clients: int, batch_size: int = 32):
    """Load MNIST dataset and create a DataLoader for a specific client."""
    transform = ToTensor()
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

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

def get_mnist_model():
    """Returns an instance of the MNIST model."""
    return Net().to(DEVICE)