import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

def split_data(dataset, num_clients, overlap_scenario=True, alpha=0.5):
    """
    Split data for label overlap or non-overlap scenarios
    overlap_scenario: True for label overlap, False for non-overlap
    alpha: Concentration parameter for Dirichlet distribution (smaller = more non-IID)
    """
    labels = np.array([label for _, label in dataset])
    num_classes = len(np.unique(labels))
    
    if overlap_scenario:
        # Label overlap scenario - clients share some classes
        label_distribution = np.random.dirichlet(np.ones(num_classes) * alpha)  # Fixed missing parenthesis
        client_indices = {i: [] for i in range(num_clients)}
        
        for class_idx in range(num_classes):
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            splits = np.array_split(class_indices, num_clients)
            for client_idx in range(num_clients):
                client_indices[client_idx].extend(splits[client_idx])
    else:
        # Label non-overlap scenario - clients have distinct classes
        client_indices = {i: [] for i in range(num_clients)}
        classes_per_client = num_classes // num_clients
        remaining_classes = num_classes % num_clients
        
        class_mappings = []
        all_classes = list(range(num_classes))
        np.random.shuffle(all_classes)
        
        for client_idx in range(num_clients):
            num_assigned = classes_per_client + (1 if client_idx < remaining_classes else 0)
            assigned_classes = all_classes[:num_assigned]
            all_classes = all_classes[num_assigned:]
            class_mappings.append(assigned_classes)
            
            for class_idx in assigned_classes:
                class_indices = np.where(labels == class_idx)[0]
                client_indices[client_idx].extend(class_indices)
    
    client_loaders = []
    for client_idx in range(num_clients):
        subset = Subset(dataset, client_indices[client_idx])
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
    
    return client_loaders, class_mappings if not overlap_scenario else None