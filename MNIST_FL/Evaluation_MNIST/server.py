import flwr as fl
from utils import evaluate_model, load_mnist
from model import get_mnist_model

def fit_config(server_round: int):
    """Return training configuration depending on the server round."""
    config = {
        "local_epochs": 2,
    }
    return config

def evaluate(server_round: int, parameters: fl.common.NDArrays, config: fl.common.Config):
    """Evaluate the global model on the entire test set."""
    model = get_mnist_model()
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    _, test_loader, _ = load_mnist(batch_size=64)
    loss, accuracy = evaluate_model(model, test_loader)
    print(f"Server-side evaluation loss {loss:.3f} / accuracy: {accuracy:.2f}%")
    return loss, {"accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.3,  # Sample 30% of available clients for each round
    min_fit_clients=2,   # Minimum number of clients to be sampled for training
    min_available_clients=10, # Minimum number of total clients in the system
    evaluate_fn=evaluate,
    on_fit_config_fn=fit_config,
)

def run_federated(num_clients=10, num_rounds=5):
    fl.simulation.start_simulation(
        #client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )