import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Paths
metadata_path = "preprocessed_data/metadata.csv"
image_dir = "preprocessed_data/images"

# Configuration
IMAGE_SIZE = (62, 48)  # Width, Height
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 100
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class CervixDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df
        self.transform = transform
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(metadata_df['class_mapped'])
        self.image_paths = metadata_df['preprocessed_image_path'].tolist()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Scale to [-1, 1]
])

# Load metadata
df = pd.read_csv(metadata_path)
df['class_mapped'] = df['class'].map({
    "Superficial-Intermediate": "Normal",
    "Parabasal": "Precancerous",
    "Koilocytotic": "Precancerous",
    "Dyskeratotic": "Cancerous",
    "Metaplastic": "Normal"
})

# Train/test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, stratify=df['class_mapped'], test_size=0.2, random_state=42)

train_loader = DataLoader(CervixDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(CervixDataset(test_df, transform), batch_size=BATCH_SIZE, shuffle=False)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(NUM_CLASSES, class_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + class_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), *self.img_shape)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, class_dim, img_shape):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, class_dim)

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape) + class_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], dim=1)
        return self.model(x)

# Models
img_shape = (3, *IMAGE_SIZE)
G = Generator(LATENT_DIM, 10, img_shape).to(DEVICE)
D = Discriminator(10, img_shape).to(DEVICE)

# Losses and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# Training Loop
for epoch in range(EPOCHS):
    G.train()
    D.train()
    for imgs, labels in train_loader:
        real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
        gen_imgs = G(z, labels)
        g_loss = criterion(D(gen_imgs, labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_imgs, labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{EPOCHS}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

# Evaluate Discriminator
D.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = D(imgs, labels)
        preds = (preds > 0.5).int().squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend([1]*len(preds))  # All are real

print("\nDiscriminator Evaluation (on real test images):")
print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))
