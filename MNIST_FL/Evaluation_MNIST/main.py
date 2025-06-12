import torch
import torch.nn.functional as F
import os
from model import get_mnist_model
from utils import load_mnist, plot_results, evaluate_model

# The train_centralized function is defined within this file
def train_centralized(model, train_loader, test_loader, epochs=10):
    optimizer = optimizer.Adam(model.parameters(), lr=0.01)
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        test_loss, accuracy = evaluate_model(model, test_loader)
        print(f"Centralized Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={accuracy:.2f}%")
        history['loss'].append(train_loss)
        history['accuracy'].append(accuracy)
    return history

# The client_fn for federated learning
import flwr as fl
from torch.utils.data import DataLoader
from model import get_mnist_model as get_federated_model # To avoid naming conflict with centralized model
from utils import load_mnist as load_federated_mnist

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader):
        self.model = get_federated_model()
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        epochs = config.get("local_epochs", 1)
        for _ in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str):
    train_loader, test_loader, _ = load_federated_mnist(batch_size=32) # Each client gets the same full dataset for simulation
    return FlowerClient(train_loader, test_loader)

# The run_federated logic is within this file
def run_federated(num_clients=10, num_rounds=5, strategy=None):
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    batch_size = 64
    train_loader, test_loader, test_dataset = load_mnist(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Centralized Learning
    centralized_model = get_mnist_model().to(device)
    print("\n--- Centralized Learning ---")
    centralized_history = train_centralized(centralized_model, train_loader, test_loader, epochs=10)
    centralized_test_loss, centralized_test_accuracy = evaluate_model(centralized_model, test_loader)
    print(f"Centralized Final Test Loss: {centralized_test_loss:.4f}, Accuracy: {centralized_test_accuracy:.2f}%")

    # Federated Learning Simulation
    print("\n--- Federated Learning Simulation ---")
    federated_history = {'loss': [], 'accuracy': []}
    def aggregate_evaluate(results):
        losses = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        return sum(losses) / total_examples, {}

    def on_fit_end(results: list[tuple[fl.client.ClientProxy, fl.common.FitRes]], failures: list[BaseException]):
        # No aggregation of fit metrics in this basic example
        pass

    def on_evaluate_end(results: list[tuple[fl.client.ClientProxy, fl.common.EvaluateRes]], failures: list[BaseException]):
        if results:
            # Aggregate evaluation metrics from clients
            total_loss = sum(res.loss * res.num_examples for _, res in results) / sum(res.num_examples for _, res in results)
            total_accuracy = sum(res.metrics["accuracy"] * res.num_examples for _, res in results) / sum(res.num_examples for _, res in results)
            federated_history['loss'].append(total_loss)
            federated_history['accuracy'].append(total_accuracy)
            print(f"Federated Round {len(federated_history['loss'])}: Aggregate Loss={total_loss:.4f}, Aggregate Accuracy={total_accuracy:.2f}%")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        min_fit_clients=2,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=aggregate_evaluate,
        on_fit_end_fn=on_fit_end,
        on_evaluate_end_fn=on_evaluate_end,
        initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in get_mnist_model().state_dict().items()]),
    )

    run_federated(num_clients=10, num_rounds=5, strategy=strategy)

    # Plotting the results
    plot_results(centralized_history, federated_history)