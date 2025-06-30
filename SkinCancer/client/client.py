import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from data_utils import load_client_data, load_metadata
from model.model import SkinCancerCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/app/data"
ALL_SCENARIOS = [
    "stat_bal_class_bal",
    "stat_bl_class_uanlba",
    "stat_unbal_class_bal",
    "stat_unbal_class_unbal"
]
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, scenario, df, label_encoder):
        self.cid = int(cid)
        self.scenario = scenario
        self.df = df
        self.label_encoder = label_encoder
        self.trainloader, self.testloader = load_client_data(
            self.cid, self.scenario, self.df, self.label_encoder,
            DATA_DIR, img_size=128, batch_size=32, num_clients=3
        )
        self.model = SkinCancerCNN(num_classes=len(self.label_encoder.classes_)).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for imgs, labels in self.trainloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate_model(self):
        """Evaluate the model and return metrics."""
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in self.testloader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return acc, prec, rec, f1

    def fit(self, parameters, config):
        """Perform one round of training."""
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the global model and return metrics."""
        self.set_parameters(parameters)
        acc, prec, rec, f1 = self.evaluate_model()
        #print(f"Client {self.cid} Scenario {self.scenario} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        print(f"\n📊 Evaluation - Client {self.cid} | Scenario: {self.scenario}")
        print(f"  - Accuracy : {acc:.4f}")
        print(f"  - Precision: {prec:.4f}")
        print(f"  - Recall   : {rec:.4f}")
        print(f"  - F1 Score : {f1:.4f}\n")
        return float(1 - acc), len(self.testloader.dataset), {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "stat_bal_class_bal"
    client_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    df = load_metadata(DATA_DIR)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["dx"])
    server_address = os.environ.get("SERVER_ADDRESS", "server:8098")
    #server_address = "server:8098"
    for scenario in ALL_SCENARIOS:
        print(f"\n=== Starting client {client_id} for scenario: {scenario} ===\n")

    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient(client_id, scenario, df, label_encoder).to_client(),
    )
