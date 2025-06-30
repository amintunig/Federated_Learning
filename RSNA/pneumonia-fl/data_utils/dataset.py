import os
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config.setting import settings

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