import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pydicom
from config.settings import settings

class FederatedPneumoniaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, client_id=None):
        """
        Args:
            df: DataFrame containing patient IDs and labels
            img_dir: Base directory containing images
            transform: Optional transform to apply
            client_id: For client-specific logging
        """
        self.df = df
        self.img_dir = img_dir
        self.client_id = client_id
        self.transform = transform or self.default_transform()
        
    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        patient_id = self.df.iloc[idx]['patientId']
        target = self.df.iloc[idx]['Target']
        
        # Try multiple image formats
        for ext in ['.png', '.jpg', '.dcm']:
            img_path = os.path.join(self.img_dir, f"{patient_id}{ext}")
            if os.path.exists(img_path):
                try:
                    if ext == '.dcm':
                        img = self.load_dicom(img_path)
                    else:
                        img = Image.open(img_path).convert('L')
                    
                    if self.transform:
                        img = self.transform(img)
                        
                    return img, torch.tensor(target, dtype=torch.float32)
                
                except Exception as e:
                    if self.client_id:
                        print(f"[Client {self.client_id}] Error loading {img_path}: {str(e)}")
                    continue
        
        # Fallback blank image
        blank = Image.new('L', (settings.IMG_SIZE, settings.IMG_SIZE))
        return self.transform(blank), torch.tensor(0, dtype=torch.float32)
    
    def load_dicom(self, path):
        """Handle DICOM files"""
        ds = pydicom.dcmread(path)
        img = Image.fromarray(ds.pixel_array).convert('L')
        return img

class DatasetPartitioner:
    def __init__(self, data_dir, num_clients=2):
        """
        Args:
            data_dir: Directory containing the dataset
            num_clients: Number of clients/partitions to create
        """
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.df = self._load_metadata()
        
    def _load_metadata(self):
        """Load and preprocess metadata CSV."""
        meta_path = os.path.join(self.data_dir, "stage2_train_metadata.csv")
        df = pd.read_csv(meta_path)
        df = df.drop_duplicates(subset=['patientId'])
        df = df.dropna(subset=['Target'])
        return df
    
    def create_partitions(self, scenario):
        """
        Create partitions based on the specified scenario.
        
        Args:
            scenario: One of ['stat_bal_class_bal', 'stat_bal_class_unbal', 
                             'stat_unbal_class_bal', 'stat_unbal_class_unbal']
        
        Returns:
            List of DataFrames for each client
        """
        if scenario == "stat_bal_class_bal":
            return self._partition_stat_balanced_class_balanced()
        elif scenario == "stat_bal_class_unbal":
            return self._partition_stat_balanced_class_unbalanced()
        elif scenario == "stat_unbal_class_bal":
            return self._partition_stat_unbalanced_class_balanced()
        elif scenario == "stat_unbal_class_unbal":
            return self._partition_stat_unbalanced_class_unbalanced()
        else:
            raise ValueError(f"Unknown scenario {scenario}")
    
    def _partition_stat_balanced_class_balanced(self):
        """Statistically balanced, class balanced partitioning (50-50% split)."""
        # Split zeros and ones separately
        zeros = self.df[self.df['Target'] == 0]
        ones = self.df[self.df['Target'] == 1]
        
        # Split zeros 50-50
        zeros_node1 = zeros.sample(frac=0.5, random_state=42)
        zeros_node2 = zeros.drop(zeros_node1.index)
        
        # Split ones 50-50
        ones_node1 = ones.sample(frac=0.5, random_state=42)
        ones_node2 = ones.drop(ones_node1.index)
        
        # Combine for each node
        node1 = pd.concat([zeros_node1, ones_node1]).sample(frac=1, random_state=42)
        node2 = pd.concat([zeros_node2, ones_node2]).sample(frac=1, random_state=42)
        
        return [node1, node2]
    
    def _partition_stat_balanced_class_unbalanced(self):
        """Statistically balanced, class unbalanced partitioning (40-60% split)."""
        # Split zeros and ones separately
        zeros = self.df[self.df['Target'] == 0]
        ones = self.df[self.df['Target'] == 1]
        
        # Split zeros 40-60
        zeros_node1 = zeros.sample(frac=0.4, random_state=42)
        zeros_node2 = zeros.drop(zeros_node1.index)
        
        # Split ones 40-60
        ones_node1 = ones.sample(frac=0.4, random_state=42)
        ones_node2 = ones.drop(ones_node1.index)
        
        # Combine for each node
        node1 = pd.concat([zeros_node1, ones_node1]).sample(frac=1, random_state=42)
        node2 = pd.concat([zeros_node2, ones_node2]).sample(frac=1, random_state=42)
        
        return [node1, node2]
    
    def _partition_stat_unbalanced_class_balanced(self):
        """Statistically unbalanced, class balanced partitioning (varying splits)."""
        partitions = []
        
        # Create splits from 10-90% to 90-10% in 10% increments
        for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            zeros = self.df[self.df['Target'] == 0]
            ones = self.df[self.df['Target'] == 1]
            
            # Split zeros
            zeros_node1 = zeros.sample(frac=split, random_state=42)
            zeros_node2 = zeros.drop(zeros_node1.index)
            
            # Split ones
            ones_node1 = ones.sample(frac=split, random_state=42)
            ones_node2 = ones.drop(ones_node1.index)
            
            # Combine for each node
            node1 = pd.concat([zeros_node1, ones_node1]).sample(frac=1, random_state=42)
            node2 = pd.concat([zeros_node2, ones_node2]).sample(frac=1, random_state=42)
            
            partitions.append((node1, node2))
        
        return partitions
    
    def _partition_stat_unbalanced_class_unbalanced(self):
        """Statistically unbalanced, class unbalanced partitioning (random splits)."""
        partitions = []
        
        # Create 5 different random splits
        for i in range(5):
            # Randomly select a split ratio for zeros and ones
            zero_split = np.random.uniform(0.1, 0.9)
            one_split = np.random.uniform(0.1, 0.9)
            
            zeros = self.df[self.df['Target'] == 0]
            ones = self.df[self.df['Target'] == 1]
            
            # Split zeros
            zeros_node1 = zeros.sample(frac=zero_split, random_state=42+i)
            zeros_node2 = zeros.drop(zeros_node1.index)
            
            # Split ones
            ones_node1 = ones.sample(frac=one_split, random_state=42+i)
            ones_node2 = ones.drop(ones_node1.index)
            
            # Combine for each node
            node1 = pd.concat([zeros_node1, ones_node1]).sample(frac=1, random_state=42+i)
            node2 = pd.concat([zeros_node2, ones_node2]).sample(frac=1, random_state=42+i)
            
            partitions.append((node1, node2))
        
        return partitions
    
    def get_dataset_stats(self, partitions):
        """Print statistics for the created partitions."""
        stats = []
        for i, (node1, node2) in enumerate(partitions):
            zeros_node1 = len(node1[node1['Target'] == 0])
            zeros_node2 = len(node2[node2['Target'] == 0])
            ones_node1 = len(node1[node1['Target'] == 1])
            ones_node2 = len(node2[node2['Target'] == 1])
            
            total_zeros = zeros_node1 + zeros_node2
            total_ones = ones_node1 + ones_node2
            
            stats.append({
                'Partition': i+1,
                'Node 1 %': f"{len(node1)/len(self.df)*100:.1f}%",
                'Node 2 %': f"{len(node2)/len(self.df)*100:.1f}%",
                'Zeros Node 1': zeros_node1,
                'Zeros Node 2': zeros_node2,
                'Ones Node 1': ones_node1,
                'Ones Node 2': ones_node2,
                'Total Zeros': total_zeros,
                'Total Ones': total_ones
            })
        
        return pd.DataFrame(stats)