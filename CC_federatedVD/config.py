import os
import torch 
from torchvision import transforms

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "preprocessed_data/metadata.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "preprocessed_data/images")

# Configuration
IMAGE_SIZE = (62, 48)  # Width, Height
BATCH_SIZE = 32
LOCAL_EPOCHS = 5  # Reduced for federated learning
GLOBAL_EPOCHS = 50
LATENT_DIM = 100
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Scale to [-1, 1]
])

# Class mappings
CLASS_MAP = {
    "Superficial-Intermediate": "Normal",
    "Parabasal": "Precancerous",
    "Koilocytotic": "Precancerous",
    "Dyskeratotic": "Cancerous",
    "Metaplastic": "Normal"
}
# Optional: Reverse mapping for label encoding
LABEL_ENCODE = {
    "Normal": 0,
    "Precancerous": 1,
    "Cancerous": 2
}