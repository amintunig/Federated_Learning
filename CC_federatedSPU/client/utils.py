import torch
import numpy as np
from typing import Dict, List, Tuple
from model.gan import Generator, Discriminator
from config import DEVICE, BATCH_SIZE
from torch.utils.data import DataLoader
from model.dataset import CervixDataset

def train_gan_epoch(
    G: Generator, 
    D: Discriminator,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    train_loader: DataLoader,
    criterion: torch.nn.Module
) -> Tuple[float, float]:
    """
    Train GAN for one epoch
    
    Returns:
        Tuple of (generator_loss, discriminator_loss)
    """
    G.train()
    D.train()
    g_losses = []
    d_losses = []
    
    for imgs, labels in train_loader:
        real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        batch_size = real_imgs.size(0)
        
        # Real and fake labels
        valid = torch.ones(batch_size, 1, device=DEVICE)
        fake = torch.zeros(batch_size, 1, device=DEVICE)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        gen_imgs = G(z, labels)
        g_loss = criterion(D(gen_imgs, labels), valid)
        g_loss.backward()
        optimizer_G.step()
        g_losses.append(g_loss.item())
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_imgs, labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        d_losses.append(d_loss.item())
    
    return np.mean(g_losses), np.mean(d_losses)

def evaluate_gan(
    G: Generator,
    D: Discriminator,
    test_loader: DataLoader,
    criterion: torch.nn.Module
) -> Dict[str, float]:
    """
    Evaluate GAN performance
    
    Returns:
        Dictionary of metrics including:
        - generator_loss
        - discriminator_loss
        - discriminator_accuracy (on real images)
    """
    G.eval()
    D.eval()
    g_losses = []
    d_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            batch_size = real_imgs.size(0)
            
            valid = torch.ones(batch_size, 1, device=DEVICE)
            fake = torch.zeros(batch_size, 1, device=DEVICE)
            
            # Generate fake images
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = G(z, labels)
            
            # Generator loss
            g_loss = criterion(D(gen_imgs, labels), valid)
            g_losses.append(g_loss.item())
            
            # Discriminator loss
            real_loss = criterion(D(real_imgs, labels), valid)
            fake_loss = criterion(D(gen_imgs, labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_losses.append(d_loss.item())
            
            # Discriminator accuracy on real images
            preds = D(real_imgs, labels)
            correct += (preds > 0.5).sum().item()
            total += batch_size
    
    return {
        "generator_loss": np.mean(g_losses),
        "discriminator_loss": np.mean(d_losses),
        "discriminator_accuracy": correct / total
    }

def get_client_data(client_id: int, batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Get data loader for specific client
    """
    dataset = CervixDataset(client_id=client_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_test_data(batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Get test data loader
    """
    dataset = CervixDataset(test=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)