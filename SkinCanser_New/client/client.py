import flwr as fl
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.model import SimpleCNN
from utils.data_utils import get_dataloader
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    loader, num_classes = get_dataloader(
        metadata_path="/app/data/metadata.csv",
        batch_size=32
    )
    return loader, num_classes

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader):
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

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
        for epoch in range(1):  # one local epoch
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true = []
        y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        if len(y_true) == 0:
            print("No data for evaluation, returning dummy metrics.")
            return float("nan"), 0, {}

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

        print(f"Evaluation metrics: {metrics}")

        return float(total_loss / len(self.trainloader)), len(self.trainloader.dataset), metrics

def main():
    trainloader, num_classes = load_data()
    model = SimpleCNN(num_classes)
    server_address = os.environ.get("SERVER_ADDRESS", "server:8085")
    #fl.client.start_client(server_address=server_address, client=FlowerClient(model, trainloader))
    fl.client.start_client(
    server_address=server_address,
    client=FlowerClient(model, trainloader).to_client()
)

if __name__ == "__main__":
    main()
