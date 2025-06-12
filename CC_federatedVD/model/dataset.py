import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader  # Fixed import
from sklearn.preprocessing import LabelEncoder
import flwr as fl

from config import METADATA_PATH, IMAGE_DIR, transform, CLASS_MAP

IMAGE_SIZE = (62, 48)
LATENT_DIM = 100

class CervixDataset(Dataset):
    def __init__(self, metadata_df=None, client_id=None, test=False):
        if metadata_df is None:
            self.metadata_df = pd.read_csv(METADATA_PATH)
            self.metadata_df['class_mapped'] = self.metadata_df['class'].map(CLASS_MAP)

            # Simulate data partitioning for federated learning
            if client_id is not None:
                all_images = self.metadata_df["preprocessed_image_path"].tolist()
                all_images.sort()  # Ensure consistent ordering
                np.random.seed(42)
                np.random.shuffle(all_images)
                # Split image paths into 3 parts
                image_splits = np.array_split(all_images, 3)
                client_images = image_splits[client_id - 1]
                # Filter metadata for this client
                self.metadata_df = self.metadata_df[self.metadata_df["preprocessed_image_path"].isin(client_images)]

            # Train/test split
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

        # Encode labels to integer
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.metadata_df['class_mapped'])

        # Clean image paths: use only the filename, handle slashes
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

        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))  # cv2 uses (width, height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
