import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_full_mimic_data():
    # Replace this with real loading of MIMIC-III dataset
    X = np.random.randn(1000, 100).astype(np.float32)
    y = np.random.randint(0, 2, 1000).astype(np.int64)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True)

def load_dummy_data():
    X = np.random.randn(100, 100).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True)
