import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np

# Define model
class MIMICModel(nn.Module):
    def __init__(self, input_dim=100, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Simulate loading MIMIC-III processed data
def load_dummy_mimic_data(input_dim=100, num_samples=1000):
    # Simulated feature matrix and binary labels
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

# Hyperparameters
input_dim = 100
num_classes = 2
batch_size = 32
epochs = 10
learning_rate = 0.001

# Load data
X, y = load_dummy_mimic_data(input_dim)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MIMICModel(input_dim=input_dim, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses, val_accuracies = [], []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    val_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
