import os
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, metadata_path, root_dir="/app", transform=None):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform

        # Use dx as class label
        self.class_names = sorted(self.df["dx"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.df["label_idx"] = self.df["dx"].map(self.class_to_idx)

        # Update filepath column to absolute paths based on mounted volumes
        self.df["full_path"] = self.df["filepath"].apply(lambda x: os.path.join("/app/data", os.path.basename(x)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        image = Image.open(img_path).convert("RGB")
        label = row["label_idx"]

        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(metadata_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = CustomImageDataset(metadata_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(dataset.class_names)

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Loss: {total_loss / len(dataloader):.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    model.train()

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
