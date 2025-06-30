import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
import torch

# ==========================
# UTILITY FUNCTION TO LOAD METADATA
# ==========================
def load_metadata(data_dir):
    # data_dir = "D:/Ascl_Mimic_Data/SkinCancerMNIST"
    """Load the HAM10000 metadata from .csv or .xlsx files."""
    csv_file = os.path.join(data_dir, "HAM10000_metadata.csv")
    xlsx_file = os.path.join(data_dir, "HAM10000_metadata.xlsx")

    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    elif os.path.exists(xlsx_file):
        return pd.read_excel(xlsx_file)
    else:
        raise FileNotFoundError(f"No HAM10000_metadata.csv or .xlsx found in {data_dir}")

# ==========================
# PARTITION SCENARIOS
# ==========================
def partition_stat_balanced_class_balanced(df, num_clients):
    dfs = [pd.DataFrame() for _ in range(num_clients)]
    for cls in df["dx"].unique():
        cls_df = df[df["dx"] == cls].sample(frac=1, random_state=42)
        splits = np.array_split(cls_df, num_clients)
        for i, split in enumerate(splits):
            dfs[i] = pd.concat([dfs[i], split])
    for i in range(num_clients):
        dfs[i] = dfs[i].sample(frac=1, random_state=42).reset_index(drop=True)
    return dfs

def partition_stat_balanced_class_unbalanced(df, num_clients):
    class_list = df["dx"].unique()
    np.random.seed(42)
    np.random.shuffle(class_list)
    class_splits = np.array_split(class_list, num_clients)
    dfs = []
    n_samples = len(df) // num_clients
    for i in range(num_clients):
        client_classes = class_splits[i]
        client_df = df[df["dx"].isin(client_classes)].sample(n=n_samples, replace=True, random_state=42)
        dfs.append(client_df.reset_index(drop=True))
    return dfs

def partition_stat_unbalanced_class_balanced(df, num_clients):
    total_samples = len(df)
    proportions = np.linspace(0.5, 1.5, num_clients)
    proportions = proportions / proportions.sum()
    client_sizes = (proportions * total_samples).astype(int)
    dfs = []
    for size in client_sizes:
        client_df = df.groupby("dx", group_keys=False).apply(
            lambda x: x.sample(int(size * len(x) / total_samples), replace=True, random_state=42)
        )
        dfs.append(client_df.sample(frac=1, random_state=42).reset_index(drop=True))
    return dfs

def partition_stat_unbalanced_class_unbalanced(df, num_clients):
    class_list = df["dx"].unique()
    dfs = []
    np.random.seed(42)
    for i in range(num_clients):
        n_classes = np.random.randint(1, len(class_list) + 1)
        client_classes = np.random.choice(class_list, n_classes, replace=False)
        n_samples = np.random.randint(100, 500)  # tune as needed
        client_df = df[df["dx"].isin(client_classes)].sample(n=n_samples, replace=True, random_state=42 + i)
        dfs.append(client_df.reset_index(drop=True))
    return dfs

# ==========================
# DATASET DEFINITION
# ==========================
class SkinCancerDataset(Dataset):
    def __init__(self, df, img_dir1, img_dir2, label_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        img_path = os.path.join(self.img_dir1, img_id + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir2, img_id + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_encoder.transform([row["dx"]])[0]
        return img, label

# ==========================
# LOAD CLIENT DATA
# ==========================
def load_client_data(client_id, scenario, df, label_encoder, data_dir, img_size=128, batch_size=32, num_clients=3):
    img_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
    img_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")

    if scenario == "stat_bal_class_bal":
        client_dfs = partition_stat_balanced_class_balanced(df, num_clients)
    elif scenario == "stat_bl_class_uanlba":
        client_dfs = partition_stat_balanced_class_unbalanced(df, num_clients)
    elif scenario == "stat_unbal_class_bal":
        client_dfs = partition_stat_unbalanced_class_balanced(df, num_clients)
    elif scenario == "stat_unbal_class_unbal":
        client_dfs = partition_stat_unbalanced_class_unbalanced(df, num_clients)
    else:
        raise ValueError(f"Unknown scenario {scenario}")

    client_df = client_dfs[client_id]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = SkinCancerDataset(client_df, img_dir1, img_dir2, label_encoder, transform)

    n = len(dataset)
    train_len = int(0.8 * n)
    test_len = n - train_len
    trainset, testset = torch.utils.data.random_split(dataset, [train_len, test_len])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader
