import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from model.model import PneumoniaCNN

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "f1": sum(f1s) / sum(examples),
    }

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        if not results:
            return None, {}

        # Aggregate loss using FedAvg's default behavior
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_loss is None:
            return None, {}

        # ✅ Unpack tuples correctly
        aggregated_metrics = weighted_average([
            (evaluate_res.num_examples, evaluate_res.metrics) for _, evaluate_res in results
        ])

        # ✅ Add per-client metrics
        for _, evaluate_res in results:
            client_id = evaluate_res.metrics.get("client_id", "unknown")
            aggregated_metrics[f"client_{client_id}_accuracy"] = evaluate_res.metrics["accuracy"]
            aggregated_metrics[f"client_{client_id}_precision"] = evaluate_res.metrics["precision"]
            aggregated_metrics[f"client_{client_id}_recall"] = evaluate_res.metrics["recall"]
            aggregated_metrics[f"client_{client_id}_f1"] = evaluate_res.metrics["f1"]

        return aggregated_loss, aggregated_metrics

def get_evaluate_fn(model):
    def evaluate_fn(server_round, parameters, config):
        model_params = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in model_params}
        model.load_state_dict(state_dict)
        return 0.0, {}  # Dummy values, real evaluation happens on clients
    return evaluate_fn

if __name__ == "__main__":
    model = PneumoniaCNN()

    strategy = AggregateCustomMetricStrategy(
        evaluate_fn=get_evaluate_fn(model),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8086",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
