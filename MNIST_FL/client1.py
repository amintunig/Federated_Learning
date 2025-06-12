import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Subset
import numpy as np

# Load dataset (non-IID split via --partition)
def load_data(partition: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # Split dataset into 3 partitions
    partition_size = len(full_dataset) // 3
    indices = list(range(len(full_dataset)))
    start = partition * partition_size
    end = start + partition_size
    subset_indices = indices[start:end]
    subset = Subset(full_dataset, subset_indices)
    return subset

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, partition: int):
        self.model = resnet50(num_classes=10)
        self.trainset = load_data(partition)
        self.trainloader = DataLoader(self.trainset, batch_size=32, shuffle=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Train for 1 epoch
            for images, labels in self.trainloader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.trainset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.trainloader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return float(loss), len(self.trainset), {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=int, required=True)
    args = parser.parse_args()

    client = FlowerClient(args.partition)
    fl.client.start_client(
        server_address="server:8085",
        client=client.to_client(),
    )

if __name__ == "__main__":
    main()
