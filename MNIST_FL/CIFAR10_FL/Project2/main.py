import sys
from client import FlowerClient
from server import start_server
import flwr as fl

if __name__ == "__main__":
    if sys.argv[1] == "server":
        start_server()
    elif sys.argv[1] == "client1":
        #fl.client.start_numpy_client("localhost:8080", client=FlowerClient(use_dummy=False))
        fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(use_dummy=False))

    elif sys.argv[1] == "client2":
        fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(use_dummy=True))
