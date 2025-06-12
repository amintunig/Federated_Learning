import flwr as fl
import argparse

from cifar_client import client_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower CIFAR-10 Client")
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="Client ID for simulation",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=3,  # Adjust as needed
        help="Total number of clients in the simulation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for local training",
    )
    args = parser.parse_args()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8088",
        client=client_fn(args.client_id, args.num_clients, args.batch_size),
    )