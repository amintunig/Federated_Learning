import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import get_mnist_model
from utils import load_mnist
from utils import evaluate_model
import torch.nn.functional as F
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader):
        self.model = get_mnist_model()
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        epochs = config.get("local_epochs", 1)
        for _ in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str):
    train_loader, test_loader, _ = load_mnist(batch_size=32) # Each client gets the same full dataset for simulation
    return FlowerClient(train_loader, test_loader)