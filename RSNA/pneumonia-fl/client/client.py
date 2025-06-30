import torch
import os
import flwr as fl
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from model.cnn import PneumoniaCNN
from model.utils import get_model_params, set_model_params
from data_utils.dataset import FederatedPneumoniaDataset
from torch.utils.data import DataLoader


class PneumoniaClient(fl.client.NumPyClient):
    def __init__(self, cid, train_df, val_df, img_dir, server_address):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PneumoniaCNN().to(self.device)
        self.cid = cid
        self.server_address = server_address

        # Balance training data
        self.train_df = self._balance_dataset(train_df)

        # Datasets and loaders
        self.train_dataset = FederatedPneumoniaDataset(self.train_df, img_dir)
        self.val_dataset = FederatedPneumoniaDataset(val_df, img_dir)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32)

        # Pos weight for BCEWithLogitsLoss
        pos = (self.train_df["Target"] == 1).sum()
        neg = (self.train_df["Target"] == 0).sum()
        self.pos_weight = torch.tensor([neg / pos]).to(self.device) if pos > 0 else torch.tensor([1.0]).to(self.device)

    def _balance_dataset(self, df):
        df_0 = df[df["Target"] == 0]
        df_1 = df[df["Target"] == 1]
        if len(df_1) == 0:
            return df  # nothing to balance
        df_0_sampled = df_0.sample(len(df_1), random_state=42)
        return pd.concat([df_0_sampled, df_1]).sample(frac=1).reset_index(drop=True)

    def get_parameters(self, config):
        return get_model_params(self.model)

    def set_parameters(self, parameters):
        set_model_params(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.pos_weight, device=self.device, epochs=1)
        return self.get_parameters(config), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = test(self.model, self.val_loader, device=self.device)
        return loss, len(self.val_dataset), metrics


def train(model, train_loader, pos_weight, device, epochs):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(model, test_loader, device):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss / len(test_loader), {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--server_address", type=str, required=True, help="Server address (e.g., 'server:8080')")
    args = parser.parse_args()

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

    client = PneumoniaClient(args.cid, train_df, val_df, args.img_dir, args.server_address)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())
