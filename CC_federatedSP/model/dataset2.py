import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import flwr as fl

from config import METADATA_PATH, IMAGE_DIR, transform, CLASS_MAP
# Additional category maps
category_map = {
    'Normal': 0,
    'Abnormal': 1,
    'Benign': 2
}

# Alternative: Map cellular types directly
Koilocytotic = 825
Dyskeratotic = 813
Metaplastic = 793
Parabasal = 789
Superficial = 831
cellular_type_mapping = {
    'Koilocytotic': 1,    # Abnormal
    'Dyskeratotic': 1,    # Abnormal
    'Metaplastic': 2,     # Benign
    'Parabasal': 0,       # Normal
    'Superficial': 0,     # Normal
}

file_paths = []
labels = []
IMAGE_SIZE = (62, 48)
LATENT_DIM = 100

def partition_data(metadata_df, num_clients=2, scenario=1, seed=42):  # Changed default to 2
    """Partition metadata DataFrame according to specified scenario.
    Returns:
        dict: {client_id: subset_of_metadata_df}
    """
    import random
    random.seed(seed)

    grouped = metadata_df.groupby('class_mapped')
    client_data = {i: pd.DataFrame() for i in range(num_clients)}

    if scenario == 1:
        for name, group in grouped:
            group = group.sample(frac=1, random_state=seed)
            split_points = np.linspace(0, len(group), num_clients + 1, dtype=int)
            for i in range(num_clients):
                client_data[i] = pd.concat([
                    client_data[i],
                    group.iloc[split_points[i]:split_points[i + 1]]
                ])

    elif scenario == 2:
        # Updated ratios for 2 clients
        class_ratios = {
            'Normal': [0.7, 0.3],
            'Abnormal': [0.4, 0.6],
            'Benign': [0.5, 0.5]
        }
        for name, group in grouped:
            group = group.sample(frac=1, random_state=seed)
            ratios = class_ratios.get(name, [1 / num_clients] * num_clients)
            split_points = np.cumsum([0] + [int(len(group) * r) for r in ratios])
            split_points[-1] = len(group)
            for i in range(num_clients):
                client_data[i] = pd.concat([
                    client_data[i],
                    group.iloc[split_points[i]:split_points[i + 1]]
                ])

    elif scenario == 3:
        # Updated ratios for 2 clients
        client_ratios = [0.6, 0.4]
        for name, group in grouped:
            group = group.sample(frac=1, random_state=seed)
            split_points = np.cumsum([0] + [int(len(group) * r) for r in client_ratios])
            split_points[-1] = len(group)
            for i in range(num_clients):
                client_data[i] = pd.concat([
                    client_data[i],
                    group.iloc[split_points[i]:split_points[i + 1]]
                ])

    elif scenario == 4:
        # Updated ratios for 2 clients
        class_client_ratios = {
            'Normal': [0.6, 0.4],
            'Abnormal': [0.3, 0.7],
            'Benign': [0.5, 0.5]
        }
        for name, group in grouped:
            group = group.sample(frac=1, random_state=seed)
            ratios = class_client_ratios.get(name, [1 / num_clients] * num_clients)
            split_points = np.cumsum([0] + [int(len(group) * r) for r in ratios])
            split_points[-1] = len(group)
            for i in range(num_clients):
                client_data[i] = pd.concat([
                    client_data[i],
                    group.iloc[split_points[i]:split_points[i + 1]]
                ])

    for i in range(num_clients):
        client_data[i] = client_data[i].sample(frac=1, random_state=seed + i)

    return client_data


class CervixDataset(Dataset):
    def __init__(self, metadata_df=None, client_id=None, test=False, scenario=1):
        if metadata_df is None:
            self.metadata_df = pd.read_csv(METADATA_PATH)
            self.metadata_df['class_mapped'] = self.metadata_df['class'].map(CLASS_MAP)

            if client_id is not None:
                client_data = partition_data(self.metadata_df, scenario=scenario)
                # Adjusted for 2 clients (0-1 index)
                self.metadata_df = client_data[client_id - 1] if client_id <= 2 else None

            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                self.metadata_df,
                stratify=self.metadata_df['class_mapped'],
                test_size=0.2,
                random_state=42
            )
            self.metadata_df = test_df if test else train_df
        else:
            self.metadata_df = metadata_df

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.metadata_df['class_mapped'])
        self.image_paths = [
            os.path.basename(str(p).replace("\\", "/")) for p in self.metadata_df['preprocessed_image_path'].tolist()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        image_path = os.path.join(IMAGE_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
