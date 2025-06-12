import argparse
import flwr as fl
import torch
from model import Net
from utils import load_cifar10, partition_data
from client import FlowerClient
from server import start_federated_server
from torch.utils.data import DataLoader

def centralized_train_and_test(batch_size, epochs, device):
    trainloader, testloader = load_cifar10(batch_size)
    model = Net().to(device)

    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    train_loss, train_correct, train_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            train_loss += criterion(preds, y).item()
            train_correct += (preds.argmax(1) == y).sum().item()
            train_total += y.size(0)

    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            test_loss += criterion(preds, y).item()
            test_correct += (preds.argmax(1) == y).sum().item()
            test_total += y.size(0)

    centralized_train_loss = train_loss / len(trainloader)
    centralized_test_loss = test_loss / len(testloader)
    centralized_train_accuracy = train_correct / train_total
    centralized_test_accuracy = test_correct / test_total

    # 🔍 Print detailed summary
    print("\n--- Centralized Training Summary ---")
    print(f"Train Loss: {centralized_train_loss:.4f}")
    print(f"Train Accuracy: {centralized_train_accuracy:.4f}")
    print(f"Test Loss: {centralized_test_loss:.4f}")
    print(f"Test Accuracy: {centralized_test_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--role", type=str, required=True, choices=["client", "server", "centralized"])
    parser.add_argument("--cid", type=int, help="Client ID if running as client")
    args = parser.parse_args()

    if args.role == "centralized":
        centralized_train_and_test(args.batch_size, args.epochs, args.device)

    elif args.role == "server":
        start_federated_server(args.num_clients, args.epochs)

    elif args.role == "client":
        full_train, full_test = load_cifar10(args.batch_size)

        client_trainsets = partition_data(full_train.dataset, args.num_clients, args.iid)
        client_testsets = partition_data(full_test.dataset, args.num_clients, args.iid)

        trainloader = DataLoader(client_trainsets[args.cid], batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(client_testsets[args.cid], batch_size=args.batch_size)

        model = Net()
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient(args.cid, model, trainloader, testloader, args.device).to_client()
        )

if __name__ == "__main__":
    main()
