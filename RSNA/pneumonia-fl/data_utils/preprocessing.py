import os
import pydicom
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


def load_metadata(data_dir):
    """Load and preprocess metadata CSV."""
    meta_path = os.path.join(data_dir, "stage2_train_metadata.csv")
    df = pd.read_csv(meta_path)

    # Basic cleaning
    df = df.drop_duplicates(subset=['patientId'])
    df = df.dropna(subset=['Target'])
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Stratified train-test split."""
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['Target'], random_state=random_state)
    return train_df, test_df


# ==========================================================
# Federated Partitioning Logic
# ==========================================================
def partition_stat_balanced_class_balanced(df, num_clients):
    """Statistically balanced, class balanced partitioning."""
    dfs = [pd.DataFrame() for _ in range(num_clients)]
    for cls in df["Target"].unique():
        cls_df = df[df["Target"] == cls].sample(frac=1, random_state=42)
        splits = np.array_split(cls_df, num_clients)
        for i, split in enumerate(splits):
            dfs[i] = pd.concat([dfs[i], split])
    for i in range(num_clients):
        dfs[i] = dfs[i].sample(frac=1, random_state=42).reset_index(drop=True)
    return dfs


def partition_stat_balanced_class_unbalanced(df, num_clients):
    """Statistically balanced, class unbalanced partitioning."""
    class_list = df["Target"].unique()
    np.random.seed(42)
    np.random.shuffle(class_list)
    class_splits = np.array_split(class_list, num_clients)
    dfs = []
    n_samples = len(df) // num_clients
    for i in range(num_clients):
        client_classes = class_splits[i]
        client_df = df[df["Target"].isin(client_classes)].sample(n=n_samples, replace=True, random_state=42)
        dfs.append(client_df.reset_index(drop=True))
    return dfs


def partition_stat_unbalanced_class_balanced(df, num_clients):
    """Statistically unbalanced, class balanced partitioning."""
    total_samples = len(df)
    proportions = np.linspace(0.5, 1.5, num_clients)
    proportions = proportions / proportions.sum()
    client_sizes = (proportions * total_samples).astype(int)

    dfs = []
    for size in client_sizes:
        client_df = df.groupby("Target", group_keys=False).apply(
            lambda x: x.sample(int(size * len(x) / total_samples), replace=True, random_state=42)
        )
        dfs.append(client_df.sample(frac=1, random_state=42).reset_index(drop=True))
    return dfs


def partition_stat_unbalanced_class_unbalanced(df, num_clients):
    """Statistically unbalanced, class unbalanced partitioning."""
    class_list = df["Target"].unique()
    dfs = []
    np.random.seed(42)
    for i in range(num_clients):
        n_classes = np.random.randint(1, len(class_list) + 1)
        client_classes = np.random.choice(class_list, n_classes, replace=False)
        n_samples = np.random.randint(100, 500)  # Adjust as needed
        client_df = df[df["Target"].isin(client_classes)].sample(n=n_samples, replace=True, random_state=42 + i)
        dfs.append(client_df.reset_index(drop=True))
    return dfs


def partition_data(df, num_clients=3, scenario="stat_bal_class_bal"):
    """Unified partition method matching data_utils.py scenarios."""
    if scenario == "stat_bal_class_bal":
        return partition_stat_balanced_class_balanced(df, num_clients)
    elif scenario == "stat_bal_class_unbal":
        return partition_stat_balanced_class_unbalanced(df, num_clients)
    elif scenario == "stat_unbal_class_bal":
        return partition_stat_unbalanced_class_balanced(df, num_clients)
    elif scenario == "stat_unbal_class_unbal":
        return partition_stat_unbalanced_class_unbalanced(df, num_clients)
    else:
        raise ValueError(f"Unknown scenario {scenario}")


# ==========================================================
# DICOM and Image Preprocessing
# ==========================================================
def dicom_to_png(dicom_path, output_dir):
    """Convert DICOM to PNG and normalize."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(dicom_path):
        if filename.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(dicom_path, filename))
            img = Image.fromarray(ds.pixel_array).convert('L')
            img.save(os.path.join(output_dir, f"{filename[:-4]}.png"))


def preprocess_image(image):
    """Normalize and standardize image."""
    image = np.array(image) / 255.0
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std
