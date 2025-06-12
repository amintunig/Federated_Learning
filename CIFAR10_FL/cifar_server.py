import flwr as fl
from typing import Callable, Dict

def test(model):
    """Centralized evaluation on the entire test set."""
    from cifar_model import get_cifar10_model, load_cifar10, DEVICE
    _, testloader = load_cifar10(0, 1)  # Load the entire test set
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return total_loss / len(testloader), accuracy

def get_evaluate_fn(model):
    """Return an evaluation function for the server."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        """Evaluate the global model on the central test set."""
        model.set_weights(parameters)
        loss, accuracy = test(model)
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 2,  # Example: set local epochs for each client
    }
    return config

def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return evaluation configuration dict for each round."""
    config = {"server_round": server_round}
    return config

if __name__ == "__main__":
    from cifar_model import get_cifar10_model
    import torch

    # Load and compile model for centralized evaluation
    central_model = get_cifar10_model()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,  # Sample 30% of available clients for each round
        fraction_evaluate=0.2,  # Sample 20% of available clients for evaluation
        min_fit_clients=3,  # Minimum number of clients to be sampled for training
        min_evaluate_clients=2,  # Minimum number of clients to be sampled for evaluation
        min_available_clients=3,  # Minimum number of total clients in the system
        #fit_config_fn=fit_config,
        #evaluate_config_fn=evaluate_config,
        evaluate_fn=get_evaluate_fn(central_model),  # Pass the central evaluation function
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:8088",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )