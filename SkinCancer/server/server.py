import flwr as fl
from model.model import SkinCancerCNN
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def get_initial_parameters():
    model = SkinCancerCNN(num_classes=7)
    return [val.cpu().numpy() for val in model.state_dict().values()]
def evaluate(self, server_round, results, failures):
    """Results contains metrics from all clients."""
    for client_result in results:
        client_id = client_result[1].cid
        metrics = client_result[1].metrics
        print(f"Round {server_round}, Client {client_id}: {metrics}")
    return super().evaluate(server_round, results, failures)
def main():
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters()),
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server(server_address="0.0.0.0:8098", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))

if __name__ == "__main__":
    main()

