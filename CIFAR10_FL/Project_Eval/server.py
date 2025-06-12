import flwr as fl

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total}

def start_federated_server(num_clients, rounds):
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
#fl.server.strategy.FedAvg