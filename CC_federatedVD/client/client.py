import argparse
import flwr as fl
import os
import torch
from torch.utils.data import DataLoader
from model.gan import get_gan_models, get_gan_optimizers
from model.dataset import CervixDataset
from config import BATCH_SIZE, LOCAL_EPOCHS, DEVICE

LATENT_DIM = 100  # Adjust if your model expects a different size

class GanClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.G, self.D = get_gan_models()
        self.G.to(DEVICE)
        self.D.to(DEVICE)
        self.optimizer_G, self.optimizer_D = get_gan_optimizers(self.G, self.D)
        self.criterion = torch.nn.BCELoss()
        
        # Use string format if your dataset expects it
        self.train_dataset = CervixDataset(client_id=cid)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            pin_memory=False  # Fixed: Set to False to avoid GPU warning
        )

    def get_parameters(self, config):
        # Concatenate Generator and Discriminator parameters
        params = [val.cpu().numpy() for val in self.G.state_dict().values()]
        params += [val.cpu().numpy() for val in self.D.state_dict().values()]
        return params

    def set_parameters(self, parameters):
        g_keys = list(self.G.state_dict().keys())
        d_keys = list(self.D.state_dict().keys())
        g_len = len(g_keys)
        g_params = parameters[:g_len]
        d_params = parameters[g_len:]

        g_state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(g_keys, g_params)}
        d_state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in zip(d_keys, d_params)}

        self.G.load_state_dict(g_state_dict, strict=True)
        self.D.load_state_dict(d_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.G.train()
        self.D.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0
        
        for epoch in range(LOCAL_EPOCHS):
            for imgs, labels in self.train_loader:
                try:
                    real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    batch_size = real_imgs.size(0)
                    
                    # Create valid and fake labels
                    valid = torch.ones(batch_size, 1, device=DEVICE, requires_grad=False)
                    fake = torch.zeros(batch_size, 1, device=DEVICE, requires_grad=False)

                    # Train Generator
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                    gen_imgs = self.G(z, labels)
                    
                    # FIXED: Ensure discriminator is called correctly
                    g_loss = self.criterion(self.D(gen_imgs, labels), valid)
                    g_loss.backward()
                    self.optimizer_G.step()

                    # Train Discriminator
                    self.optimizer_D.zero_grad()
                    
                    # FIXED: Make sure real_imgs and labels are compatible
                    real_pred = self.D(real_imgs, labels)
                    real_loss = self.criterion(real_pred, valid)
                    
                    # Generate fake images and get discriminator prediction
                    with torch.no_grad():
                        fake_imgs = self.G(z, labels)
                    fake_pred = self.D(fake_imgs.detach(), labels)
                    fake_loss = self.criterion(fake_pred, fake)
                    
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    print(f"real_imgs shape: {real_imgs.shape if 'real_imgs' in locals() else 'undefined'}")
                    print(f"labels shape: {labels.shape if 'labels' in locals() else 'undefined'}")
                    print(f"labels dtype: {labels.dtype if 'labels' in locals() else 'undefined'}")
                    raise e
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Return metrics
        avg_g_loss = total_g_loss / max(num_batches, 1)
        avg_d_loss = total_d_loss / max(num_batches, 1)
        
        return (
            self.get_parameters(config={}), 
            len(self.train_dataset), 
            {"generator_loss": avg_g_loss, "discriminator_loss": avg_d_loss}
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.G.eval()
        self.D.eval()
        
        # Simple evaluation: compute discriminator accuracy on real vs fake
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for imgs, labels in self.train_loader:
                real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                batch_size = real_imgs.size(0)
                
                # Test on real images
                real_pred = self.D(real_imgs, labels)
                real_correct = (real_pred > 0.5).sum().item()
                
                # Test on fake images
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_imgs = self.G(z, labels)
                fake_pred = self.D(fake_imgs, labels)
                fake_correct = (fake_pred <= 0.5).sum().item()
                
                total_correct += real_correct + fake_correct
                total_samples += batch_size * 2  # Real + fake
        
        accuracy = total_correct / max(total_samples, 1)
        return 0.0, len(self.train_dataset), {"accuracy": accuracy}

def main():
    """Starts a Flower client using the updated API."""
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (e.g., 1, 2, 3)")
    args = parser.parse_args()

    # Get the server address from the environment variable set in docker-compose
    server_address = os.getenv("FL_SERVER_ADDRESS", "127.0.0.1:8084")

    # FIXED: Instantiate the client and convert it to the new Client type
    numpy_client = GanClient(cid=args.cid)
    client = numpy_client.to_client()  # This fixes the deprecation warning

    # Start the client using the recommended function
    fl.client.start_client(
        server_address=server_address,
        client=client,
        insecure=True  # Add this for Docker environments
    )

if __name__ == "__main__":
    main()