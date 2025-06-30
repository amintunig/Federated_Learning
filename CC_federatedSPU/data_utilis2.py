import os
import random
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def load_sipakmed_data(data_dir):
    """
    Load SIPaKMeD dataset from directory structure.
    Args:
        data_dir (str): Path to dataset root directory
    Returns:
        tuple: (list of file paths, list of integer labels)
    """
    classes = ["Normal", "Abnormal", "Benign"]
    file_paths, labels = [], []
    # cellular_type_mapping = {
    #     'Koilocytotic': 1,    # Abnormal
    #     'Dyskeratotic': 1,    # Abnormal
    #     'Metaplastic': 2,     # Benign
    #     'Parabasal': 0,       # Normal
    #     'Superficial': 0,     # Normal
    # }
    
    # Walk through directory structure
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Collect all valid image files in class directory
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    file_paths.append(os.path.join(root, fname))
                    labels.append(label_idx)
    
    return file_paths, labels

def partition_data(file_paths, labels, num_clients=2, scenario=1, seed=42):
    """
    Partition data into clients based on specified scenario.
    Args:
        file_paths (list): List of image file paths
        labels (list): Corresponding integer labels
        num_clients (int): Number of clients (default=2)
        scenario (int): Partitioning scenario (1-4)
        seed (int): Random seed for reproducibility
    Returns:
        dict: {client_id: (file_paths, labels)}
    """
    random.seed(seed)
    
    # Group indices by class
    indices_by_class = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)
    
    # Shuffle indices within each class
    for idxs in indices_by_class.values():
        random.shuffle(idxs)
    
    # Initialize client data storage
    client_data = {i: {'file_paths': [], 'labels': []} for i in range(num_clients)}
    
    # Scenario 1: Statistically balanced, class balanced (50-50 split)
    if scenario == 1:
        for cls, idxs in indices_by_class.items():
            split_point = len(idxs) // 2
            # First half to client 0
            for idx in idxs[:split_point]:
                client_data[0]['file_paths'].append(file_paths[idx])
                client_data[0]['labels'].append(labels[idx])
            # Second half to client 1
            for idx in idxs[split_point:]:
                client_data[1]['file_paths'].append(file_paths[idx])
                client_data[1]['labels'].append(labels[idx])
    
    # Scenario 2: Statistically unbalanced, class balanced (30-70 split)
    elif scenario == 2:
        for cls, idxs in indices_by_class.items():
            split_point = int(len(idxs) * 0.3)
            # 30% to client 0
            for idx in idxs[:split_point]:
                client_data[0]['file_paths'].append(file_paths[idx])
                client_data[0]['labels'].append(labels[idx])
            # 70% to client 1
            for idx in idxs[split_point:]:
                client_data[1]['file_paths'].append(file_paths[idx])
                client_data[1]['labels'].append(labels[idx])
    
    # Scenario 3: Statistically balanced, class unbalanced
    elif scenario == 3:
        class_ratios = {
            0: 0.75,  # Normal: 75% to client 0
            1: 0.25,  # Abnormal: 25% to client 0
            2: 0.5    # Benign: 50% to client 0
        }
        for cls, idxs in indices_by_class.items():
            split_point = int(len(idxs) * class_ratios.get(cls, 0.5))
            for idx in idxs[:split_point]:
                client_data[0]['file_paths'].append(file_paths[idx])
                client_data[0]['labels'].append(labels[idx])
            for idx in idxs[split_point:]:
                client_data[1]['file_paths'].append(file_paths[idx])
                client_data[1]['labels'].append(labels[idx])
    
    # Scenario 4: Statistically unbalanced, class unbalanced
    elif scenario == 4:
        class_ratios = {
            0: 0.5,   # Normal: 50% to client 0
            1: 0.15,  # Abnormal: 15% to client 0
            2: 0.33   # Benign: 33% to client 0
        }
        for cls, idxs in indices_by_class.items():
            split_point = int(len(idxs) * class_ratios.get(cls, 0.5))
            for idx in idxs[:split_point]:
                client_data[0]['file_paths'].append(file_paths[idx])
                client_data[0]['labels'].append(labels[idx])
            for idx in idxs[split_point:]:
                client_data[1]['file_paths'].append(file_paths[idx])
                client_data[1]['labels'].append(labels[idx])
    
    # Shuffle each client's data
    for i in range(num_clients):
        combined = list(zip(client_data[i]['file_paths'], client_data[i]['labels']))
        random.shuffle(combined)
        if combined:
            paths, labs = zip(*combined)
            client_data[i]['file_paths'], client_data[i]['labels'] = list(paths), list(labs)
        else:
            client_data[i]['file_paths'], client_data[i]['labels'] = [], []
    
    return {i: (client_data[i]['file_paths'], client_data[i]['labels']) for i in range(num_clients)}

class SipaKMedDataset(Dataset):
    """PyTorch Dataset for SIPaKMeD images with preprocessing."""
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of image paths
            labels (list): Integer class labels
            transform (callable): Optional transform to apply
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Standard size for CNNs
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Load image with OpenCV (consistent with your other code)
            img = cv2.imread(self.file_paths[idx])
            if img is None:
                raise FileNotFoundError(f"Image not found: {self.file_paths[idx]}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            label = self.labels[idx]
            return img, label
        
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {str(e)}")
            # Return a zero tensor if image fails to load
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, 0  # Default to class 0

def get_client_data(client_id, data_dir, scenario=1):
    """
    Helper function to load and partition data for a specific client.
    Args:
        client_id (int): Client ID (0 or 1)
        data_dir (str): Path to dataset root
        scenario (int): Partitioning scenario (1-4)
    Returns:
        SipaKMedDataset: Dataset for the specified client
    """
    file_paths, labels = load_sipakmed_data(data_dir)
    client_data = partition_data(file_paths, labels, scenario=scenario)
    client_files, client_labels = client_data[client_id]
    return SipaKMedDataset(client_files, client_labels)

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/sipakmed"
    file_paths, labels = load_sipakmed_data(data_dir)
    print(f"Total images loaded: {len(file_paths)}")
    
    # Test all scenarios
    for scenario in range(1, 5):
        client_data = partition_data(file_paths, labels, scenario=scenario)
        print(f"\nScenario {scenario} distribution:")
        for client_id, (paths, labs) in client_data.items():
            print(f"Client {client_id}: {len(paths)} images")
            print(f"  Class counts: {np.bincount(labs)}")