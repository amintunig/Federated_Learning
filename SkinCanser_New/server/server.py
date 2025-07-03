# import flwr as fl
# import torch
# from model.model import SimpleCNN
# from utils.data_utils import get_dataloader


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_evaluate_fn():
#     testloader, _ = get_dataloader("/data", batch_size=32)
#     def evaluate(server_round, parameters, config):
#         model = SimpleCNN()
#         model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
#         model.to(DEVICE)
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for images, labels in testloader:
#                 images, labels = images.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         accuracy = correct / total
#         return float(0), {"accuracy": accuracy}
#     return evaluate

# def main():
#     strategy = fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn())
#     fl.server.start_server(server_address="0.0.0.0:8085", strategy=strategy)

import flwr as fl
import torch
from model.model import SimpleCNN
from utils.data_utils import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weighted_average(metrics_list):
    total = 0
    aggregated = {}
    for tup in metrics_list:
        if len(tup) != 3:
            print(f"Skipping aggregation because of wrong tuple size: {len(tup)}")
            continue
        _, num_examples, metrics = tup
        total += num_examples
        for k, v in metrics.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * num_examples

    if total == 0:
        return {}

    # divide by total to get weighted average
    for k in aggregated:
        aggregated[k] /= total

    return aggregated



def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
