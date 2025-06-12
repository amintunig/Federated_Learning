import flwr as fl

# Updated server configuration with proper strategy
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,  # Minimum clients needed to start training
    min_fit_clients=2,        # Minimum clients required for each round
    min_evaluate_clients=2,   # Minimum clients for evaluation
    fraction_fit=1.0,         # Use 100% of available clients for training
    fraction_evaluate=1.0,    # Use 100% for evaluation
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),  # Proper config format
    strategy=strategy
)
