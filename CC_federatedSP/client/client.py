import argparse
import flwr as fl
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.gan import get_gan_models, get_gan_optimizers
from model.dataset2 import partition_data, CervixDataset
from config import BATCH_SIZE, LOCAL_EPOCHS, DEVICE, CLASS_MAP

LATENT_DIM = 100  # Adjust if your model expects a different size

class GanClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.G, self.D = get_gan_models()
        self.G.to(DEVICE)
        self.D.to(DEVICE)
        self.optimizer_G, self.optimizer_D = get_gan_optimizers(self.G, self.D)
        self.criterion = torch.nn.BCELoss()

        # Load full metadata
        import pandas as pd
        from config import METADATA_PATH

        metadata_df = pd.read_csv(METADATA_PATH)
        metadata_df['class_mapped'] = metadata_df['class'].map(CLASS_MAP)

        # Partition data for 2 clients (adjust scenario as needed)
        client_data_dict = partition_data(metadata_df, num_clients=2, scenario=1, seed=42)

        # Get this client's data subset (client_id 1 or 2)
        if cid not in [1, 2]:
            raise ValueError("Client ID must be 1 or 2 for this setup.")
        client_metadata = client_data_dict[cid - 1]

        # Create dataset and dataloader
        self.train_dataset = CervixDataset(metadata_df=client_metadata, test=False, scenario=1)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # Shuffle training data
            pin_memory=torch.cuda.is_available()
        )

    def get_parameters(self, config):
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

                    valid = torch.ones(batch_size, 1, device=DEVICE, requires_grad=False)
                    fake = torch.zeros(batch_size, 1, device=DEVICE, requires_grad=False)

                    # Train Generator
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                    gen_imgs = self.G(z, labels)

                    g_loss = self.criterion(self.D(gen_imgs, labels), valid)
                    g_loss.backward()
                    self.optimizer_G.step()

                    # Train Discriminator
                    self.optimizer_D.zero_grad()

                    real_pred = self.D(real_imgs, labels)
                    real_loss = self.criterion(real_pred, valid)

                    fake_pred = self.D(gen_imgs.detach(), labels)
                    fake_loss = self.criterion(fake_pred, fake)

                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    self.optimizer_D.step()

                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    raise e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        y_true = []
        y_pred = []

        with torch.no_grad():
            for imgs, labels in self.train_loader:
                real_imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                batch_size = real_imgs.size(0)

                real_pred = self.D(real_imgs, labels)
                real_labels = torch.ones(batch_size, device=DEVICE)

                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_imgs = self.G(z, labels)
                fake_pred = self.D(fake_imgs, labels)
                fake_labels = torch.zeros(batch_size, device=DEVICE)

                y_true.extend(real_labels.cpu().numpy())
                y_pred.extend((real_pred > 0.5).cpu().numpy())

                y_true.extend(fake_labels.cpu().numpy())
                y_pred.extend((fake_pred > 0.5).cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return 0.0, len(self.train_dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (1 or 2)")
    args = parser.parse_args()

    server_address = os.getenv("FL_SERVER_ADDRESS", "127.0.0.1:8084")

    numpy_client = GanClient(cid=args.cid)
    client = numpy_client.to_client()  # Adapted for Flower API

    fl.client.start_client(
        server_address=server_address,
        client=client,
        insecure=True  # Useful for Docker environments
    )


if __name__ == "__main__":
    main()
