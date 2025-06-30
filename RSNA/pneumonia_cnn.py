import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Configuration
DATA_DIR = "D:/Ascl_Mimic_Data/RSNA"  # Changed from "rsna-pneumonia-detection-challenge"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "Training/Images")  # Changed from "stage_2_train_images"
TEST_IMG_DIR = os.path.join(DATA_DIR, "Test")  # Changed from "stage_2_test_images"
TRAIN_META_PATH = os.path.join(DATA_DIR, "stage2_train_metadata.csv") 
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
print(DATA_DIR)

# Verify paths exist
print(f"Checking paths:")
print(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")
print(f"TRAIN_IMG_DIR exists: {os.path.exists(TRAIN_IMG_DIR)}")
print(f"TEST_IMG_DIR exists: {os.path.exists(TEST_IMG_DIR)}")
print(f"TRAIN_META_PATH exists: {os.path.exists(TRAIN_META_PATH)}")
# Load metadata
# Load metadata - Updated to match your file names
train_meta = pd.read_csv(TRAIN_META_PATH)  # Changed from "stage_2_train_labels.csv"
#os.path.join(DATA_DIR, "stage2_train_metadata.csv")
class_counts = train_meta['Target'].value_counts()
print("Class distribution:\n", class_counts)
# Verify the first few image paths
print("\nSample image paths from metadata:")
print(train_meta.head()[['patientId', 'Target']])
# # Custom Dataset
class PneumoniaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        #Get patient ID and construct correct image path
        patient_id = self.df.iloc[idx]['patientId']
        
        # Try different image extensions if needed
        for ext in ['.png', '.jpg', '.jpeg', '.dcm']:
            img_path = os.path.join(self.img_dir, f"{patient_id}{ext}")
            if os.path.exists(img_path):
                try:
                    if ext == '.dcm':
                        # Handle DICOM files
                        import pydicom
                        ds = pydicom.dcmread(img_path)
                        image = Image.fromarray(ds.pixel_array).convert('L')
                    else:
                        # Handle regular image files
                        image = Image.open(img_path).convert('L')
                    
                    if self.transform:
                        image = self.transform(image)
                        
                    label = self.df.iloc[idx]['Target']
                    return image, torch.tensor(label, dtype=torch.float32)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    continue
        
        # If no image found, return a blank image and print warning
        print(f"Warning: No image found for {patient_id} in {self.img_dir}")
        blank_image = Image.new('L', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            blank_image = self.transform(blank_image)
        return blank_image, torch.tensor(0, dtype=torch.float32)
# # Simple CNN Model
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (IMG_SIZE//4) * (IMG_SIZE//4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * (IMG_SIZE//4) * (IMG_SIZE//4))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

# Prepare data
train_df, val_df = train_test_split(train_meta, test_size=0.2, stratify=train_meta['Target'], random_state=42)

train_dataset = PneumoniaDataset(train_df, TRAIN_IMG_DIR)
val_dataset = PneumoniaDataset(val_df, TRAIN_IMG_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
val_losses = []
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs.squeeze(), labels).item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Store metrics
    epoch_train_loss = running_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Accuracy: {accuracy:.4f} | "
          f"Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | "
          f"F1: {f1:.4f}")

# Plotting
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Metrics plot
plt.subplot(1, 2, 2)
for metric, values in metrics.items():
    plt.plot(values, label=metric)
plt.title('Validation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()

# Save model
torch.save(model.state_dict(), 'pneumonia_cnn.pth')
print("Training complete. Model saved.")