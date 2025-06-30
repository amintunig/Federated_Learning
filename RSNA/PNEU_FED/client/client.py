import argparse
import flwr as fl
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import FederatedPneumoniaDataset
from model.utils import get_model_params, set_model_params
from model.model import PneumoniaCNN  # Ensure you have a valid model defined here

class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, model, train_df, test_df, img_dir, client_id):
        self.model = model
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = FederatedPneumoniaDataset(train_df, img_dir, client_id=client_id)
        self.test_dataset = FederatedPneumoniaDataset(test_df, img_dir, client_id=client_id)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32)

        targets = train_df['Target'].values
        self.pos_weight = torch.tensor([len(targets) / sum(targets) - 1], device=self.device)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.pos_weight, device=self.device, epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, metrics = test(self.model, self.test_loader, self.pos_weight, device=self.device)
        return float(loss), len(self.test_loader.dataset), {
            "accuracy": float(metrics['accuracy']),
            "precision": float(metrics['precision']),
            "recall": float(metrics['recall']),
            "f1": float(metrics['f1']),
            "client_id": self.client_id
        }

def train(model, train_loader, pos_weight, device, epochs=1):
    model.train()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, test_loader, pos_weight, device):
    model.eval()
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    avg_loss = total_loss / len(test_loader)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return avg_loss, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--server_address", type=str, default="localhost:8086")
    # parser.add_argument("--client_id", type=int, required=True)
    # parser.add_argument("--img_dir", type=str, default="/data/images")
    # parser.add_argument("--metadata_path", type=str, default="/data/stage2_train_metadata.csv")
    # args = parser.parse_args()

    # # Load and prepare data
    # train_df = pd.read_csv(args.train_file)
    # val_df = pd.read_csv(args.val_file)

    # # Clean column names
    # train_df.columns = train_df.columns.str.strip()
    # val_df.columns = val_df.columns.str.strip()

    # # Rename if needed
    # if "PredictionString" in val_df.columns:
    #     val_df.rename(columns={"PredictionString": "Target"}, inplace=True)

    # # Ensure 'Target' column exists
    # assert "Target" in train_df.columns, "Train CSV missing 'Target' column"
    # assert "Target" in val_df.columns, "Validation CSV missing 'Target' column"

    # client = PneumoniaClient(args.cid, train_df, val_df, args.img_dir, args.server_address)
    # fl.client.start_client(server_address=args.server_address, client=client.to_client())
    import argparse
# ... (other imports)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="server:8086")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    args = parser.parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load metadata
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    # Clean column names
    train_df.columns = train_df.columns.str.strip()
    val_df.columns = val_df.columns.str.strip()

    # Rename if needed
    if "PredictionString" in val_df.columns:
        val_df.rename(columns={"PredictionString": "Target"}, inplace=True)

    # Ensure 'Target' column exists
    assert "Target" in train_df.columns, "Train CSV missing 'Target' column"
    assert "Target" in val_df.columns, "Validation CSV missing 'Target' column"

    # Initialize model and client
    model = PneumoniaCNN()
    client = PneumoniaClient(model, train_df, val_df, args.img_dir, args.client_id)

    # Start client
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


    


    # model = PneumoniaCNN()
    # client = PneumoniaClient(model, train_df, test_df, args.img_dir, args.client_id)

    # # Start Flower client
    # fl.client.start_client(server_address=args.server_address, client=client)
