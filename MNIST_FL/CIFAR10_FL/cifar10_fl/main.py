import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from model import CIFAR10Model
from server import Server
from client import Client
from utils import split_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.995)
    parser.add_argument('--overlap', type=bool, default=True)
    args = parser.parse_args()
    
    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Split data according to scenario
    client_loaders, _ = split_data(
        train_dataset, args.num_clients, overlap_scenario=args.overlap)
    
    # Initialize model and clients
    model = CIFAR10Model()
    clients = [Client(model, loader, args, i) for i, loader in enumerate(client_loaders)]
    
    # Initialize and run server
    server = Server(model, clients, test_loader, args)
    server.run_experiment()
    
    # Print final results in table format
    print("\nFinal Results:")
    print("t\ttn\tsolo\t25%\t50%\t75%")
    for i, round in enumerate(server.results['rounds']):
        print(f"{round}\t"
              f"{server.results['solo_acc'][i]:.2f}\t"
              f"{server.results['fed_acc_25'][i]:.2f}\t"
              f"{server.results['fed_acc_50'][i]:.2f}\t"
              f"{server.results['fed_acc_75'][i]:.2f}")

if __name__ == '__main__':
    main()