import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import OrderedDict

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset for non-IID data
class NonIIDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Federated Learning Client
class FLClient:
    def __init__(self, model, data, labels):
        self.model = model
        self.dataset = NonIIDDataset(data, labels)
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for data, labels in self.loader:
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()

# Federated Learning Server
class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.clients = []

    def add_client(self, client):
        self.clients.append(client)

    def aggregate(self, client_weights):
        global_weights = OrderedDict()
        for key in client_weights[0].keys():
            global_weights[key] = torch.stack([weights[key] for weights in client_weights]).mean(0)
        self.global_model.load_state_dict(global_weights)
        return global_weights

    def run(self, rounds=10, epochs_per_client=1):
        for _ in range(rounds):
            client_weights = []
            for client in self.clients:
                weights = client.train(epochs_per_client)
                client_weights.append(weights)
            global_weights = self.aggregate(client_weights)
            print(f"Round {_ + 1}: Global model updated.")

# Example usage
if __name__ == "__main__":
    # Generate non-IID data for clients
    num_clients = 5
    data_size = 100
    input_dim = 10
    output_dim = 2

    clients_data = []
    clients_labels = []
    for _ in range(num_clients):
        # Simulate non-IID data by skewing label distributions
        skew = np.random.randint(0, output_dim)
        data = torch.randn(data_size, input_dim)
        labels = torch.randint(skew, skew + 1, (data_size,)) % output_dim
        clients_data.append(data)
        clients_labels.append(labels)

    # Initialize global model and server
    global_model = SimpleNN()
    server = FLServer(global_model)

    # Add clients with non-IID data
    for i in range(num_clients):
        client_model = SimpleNN()
        client = FLClient(client_model, clients_data[i], clients_labels[i])
        server.add_client(client)

    # Run federated learning
    server.run(rounds=5, epochs_per_client=2)