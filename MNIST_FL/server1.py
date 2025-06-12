import flwr as fl
from typing import Dict, List, Optional, Tuple

class CheckpointStrategy(fl.server.strategy.FedAvg):
    def __init__(self, total_epochs: int, checkpoint_epoch: float = 0.75, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_epoch = int(total_epochs * checkpoint_epoch)
        self.centralized_accuracy = 0.85  # Replace with actual centralized accuracy

    def aggregate_fit(self, rnd: int, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if rnd == self.checkpoint_epoch:
            print(f"\n[CHECKPOINT] Round {rnd}: Comparing federated vs. centralized...")
            fed_accuracy = sum([res.metrics["accuracy"] for _, res in results]) / len(results)
            print(f"Federated Accuracy: {fed_accuracy:.2f} | Centralized: {self.centralized_accuracy:.2f}")
        return aggregated

def main():
    strategy = CheckpointStrategy(
        total_epochs=20,
        checkpoint_epoch=0.75,
        min_fit_clients=3,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8084",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
