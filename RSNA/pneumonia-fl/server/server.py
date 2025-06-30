import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays
from model.cnn import PneumoniaCNN
from model.utils import set_model_params


# Aggregates evaluation metrics from clients using weighted average
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    def weighted(metric_name):
        return sum(m[1][metric_name] * m[0] for m in metrics) / total_examples

    return {
        "accuracy": weighted("accuracy"),
        "precision": weighted("precision"),
        "recall": weighted("recall"),
        "f1_score": weighted("f1_score"),
    }


# Custom FedAvg strategy with model saving after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters_tuple is not None:
            aggregated_parameters, _ = aggregated_parameters_tuple

            # Save model
            model = PneumoniaCNN()
            set_model_params(model, parameters_to_ndarrays(aggregated_parameters))
            torch.save(model.state_dict(), f"global_model_round_{rnd}.pth")
            print(f"[ROUND {rnd}] ✅ Global model saved to 'global_model_round_{rnd}.pth'")

        return aggregated_parameters_tuple


# Starts the Flower server
def start_server(num_rounds=3):
    strategy = SaveModelStrategy(
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
