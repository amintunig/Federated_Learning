import flwr as fl
import torch
from model import Net
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
    # Compute training accuracy/loss after final epoch
    net.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            preds = net(x)
            loss_total += criterion(preds, y).item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_total / len(trainloader), correct / total

def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = net(x)
            loss_total += criterion(preds, y).item()
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_total / len(testloader), correct / total

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader, device):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config: Dict[str, str]) -> List:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List, config: Dict[str, str]) -> Tuple[List, int, Dict]:
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 1))
        train_loss, train_accuracy = train(self.model, self.trainloader, local_epochs, self.device)
        print(f"[Client {self.cid}] Local Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"loss": train_loss, "accuracy": train_accuracy},
        )

    def evaluate(self, parameters: List, config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        print(f"[Client {self.cid}] Local Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
