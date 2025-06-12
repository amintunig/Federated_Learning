import torch
import numpy as np
from typing import Dict, List, Tuple
from model.gan import Generator, Discriminator
from config import DEVICE

def aggregate_gan_parameters(results: List[Tuple[np.ndarray, int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Aggregate GAN parameters from multiple clients using weighted average
    
    Args:
        results: List of tuples containing (client_parameters, num_examples)
                where client_parameters is [G_params, D_params]
    
    Returns:
        Tuple of (aggregated_G_params, aggregated_D_params)
    """
    # Initialize accumulators
    total_examples = sum([num_examples for _, num_examples in results])
    G_accumulator = None
    D_accumulator = None
    
    for (G_params, D_params), num_examples in results:
        # Convert numpy arrays to torch tensors for proper weighted averaging
        G_tensors = [torch.from_numpy(arr) for arr in G_params]
        D_tensors = [torch.from_numpy(arr) for arr in D_params]
        
        # Initialize accumulators if needed
        if G_accumulator is None:
            G_accumulator = [torch.zeros_like(t) for t in G_tensors]
            D_accumulator = [torch.zeros_like(t) for t in D_tensors]
        
        # Weighted sum
        weight = num_examples / total_examples
        for i in range(len(G_accumulator)):
            G_accumulator[i] += G_tensors[i] * weight
            D_accumulator[i] += D_tensors[i] * weight
    
    # Convert back to numpy arrays
    aggregated_G = [t.numpy() for t in G_accumulator]
    aggregated_D = [t.numpy() for t in D_accumulator]
    
    return aggregated_G, aggregated_D

def save_gan_models(G: Generator, D: Discriminator, path: str = "saved_models"):
    """
    Save generator and discriminator models to disk
    """
    torch.save(G.state_dict(), f"{path}/generator.pth")
    torch.save(D.state_dict(), f"{path}/discriminator.pth")

def load_gan_models(G: Generator, D: Discriminator, path: str = "saved_models"):
    """
    Load generator and discriminator models from disk
    """
    G.load_state_dict(torch.load(f"{path}/generator.pth", map_location=DEVICE))
    D.load_state_dict(torch.load(f"{path}/discriminator.pth", map_location=DEVICE))
    return G, D

def get_model_parameters(G: Generator, D: Discriminator) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract parameters from models as numpy arrays
    """
    G_params = [val.cpu().numpy() for _, val in G.state_dict().items()]
    D_params = [val.cpu().numpy() for _, val in D.state_dict().items()]
    return G_params, D_params

def set_model_parameters(G: Generator, D: Discriminator, G_params: List[np.ndarray], D_params: List[np.ndarray]):
    """
    Set model parameters from numpy arrays
    """
    G_state_dict = {k: torch.tensor(v) for k, v in zip(G.state_dict().keys(), G_params)}
    D_state_dict = {k: torch.tensor(v) for k, v in zip(D.state_dict().keys(), D_params)}
    
    G.load_state_dict(G_state_dict, strict=True)
    D.load_state_dict(D_state_dict, strict=True)
    return G, D