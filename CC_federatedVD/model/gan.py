import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import LATENT_DIM, NUM_CLASSES, DEVICE, IMAGE_SIZE

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, class_dim=10, img_shape=(3, *IMAGE_SIZE)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(NUM_CLASSES, class_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + class_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        """
        Forward pass for Generator
        Args:
            z: noise vector of shape (batch_size, latent_dim)
            labels: class labels of shape (batch_size,)
        Returns:
            generated images of shape (batch_size, *img_shape)
        """
        # Ensure labels are long type for embedding
        if labels.dtype != torch.long:
            labels = labels.long()
        
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, class_dim=10, img_shape=(3, *IMAGE_SIZE)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(NUM_CLASSES, class_dim)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + class_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        """
        Forward pass for Discriminator
        Args:
            img: input images of shape (batch_size, *img_shape)
            labels: class labels of shape (batch_size,)
        Returns:
            predictions of shape (batch_size, 1)
        """
        # Ensure labels are long type for embedding
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Get label embedding
        c = self.label_emb(labels)
        
        # Concatenate flattened image and label embedding
        x = torch.cat([img_flat, c], dim=1)
        
        return self.model(x)

def get_gan_models():
    """Initialize and return Generator and Discriminator models"""
    img_shape = (3, *IMAGE_SIZE)
    G = Generator(LATENT_DIM, 10, img_shape)
    D = Discriminator(10, img_shape)
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    
    G.apply(weights_init)
    D.apply(weights_init)
    
    return G.to(DEVICE), D.to(DEVICE)

def get_gan_optimizers(G, D):
    """Initialize and return optimizers for Generator and Discriminator"""
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return optimizer_G, optimizer_D