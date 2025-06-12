import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def partition_data(dataset, num_clients, iid=False):
    labels = np.array(dataset.targets)
    if iid:
        num_items = len(dataset) // num_clients
        return [torch.utils.data.Subset(dataset, list(range(i * num_items, (i + 1) * num_items)))
                for i in range(num_clients)]
    else:
        num_classes = 10
        alpha = 0.5
        client_indices = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            proportions = np.random.dirichlet(alpha=np.ones(num_clients))
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] += len(class_indices) - np.sum(proportions)
            class_indices = np.random.permutation(class_indices)
            offset = 0
            for cid in range(num_clients):
                client_indices[cid].extend(class_indices[offset:offset + proportions[cid]])
                offset += proportions[cid]
        return [torch.utils.data.Subset(dataset, idx) for idx in client_indices]
