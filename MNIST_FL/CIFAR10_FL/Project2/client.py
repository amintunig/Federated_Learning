import flwr as fl
import torch
from model import MIMICModel
from utils import load_full_mimic_data, load_dummy_data
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, use_dummy=False):
        self.model = MIMICModel().to(DEVICE)
        self.trainloader = load_dummy_data() if use_dummy else load_full_mimic_data()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(1):  # 1 local epoch
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, loss = 0, 0
        total = 0
        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = self.model(x)
                loss += F.cross_entropy(output, y, reduction='sum').item()
                pred = output.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return float(loss / total), total, {"accuracy": accuracy}
