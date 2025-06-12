import flwr as fl
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from model import Net, load_mnist, DEVICE

class MnistClient(fl.client.NumPyClient):
    """Flower client for MNIST."""

    def __init__(self, client_id: int, num_clients: int, batch_size: int):
        self.client_id = client_id
        self.trainloader, self.testloader = load_mnist(client_id, num_clients, batch_size)
        self.model = Net().to(DEVICE)

    def get_parameters(self, config: Dict[str, fl.common.NDArrays]) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[fl.common.NDArrays, int, Dict[str, fl.common.Scalar]]:
        """Train the model on the locally held dataset."""
        self.set_parameters(parameters)
        epochs = config.get("epochs", 1)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.trainloader.dataset), {"loss": loss.item()}

    def evaluate(
        self,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate the model on the locally held dataset."""
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return total_loss / len(self.testloader), len(self.testloader.dataset), {"accuracy": accuracy}

def client_fn(client_id: int, num_clients: int, batch_size: int):
    """Creates a Flower client for MNIST."""
    return MnistClient(client_id, num_clients, batch_size)