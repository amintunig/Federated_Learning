import torch
import numpy as np
from collections import OrderedDict

class Server:
    def __init__(self, model, clients, test_loader, args):
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.args = args
        self.global_weights = model.state_dict()
        self.results = {
            'rounds': [],
            'solo_acc': [],
            'fed_acc_25': [],
            'fed_acc_50': [],
            'fed_acc_75': []
        }
        
    def aggregate(self, client_weights, participation_rate):
        """FedAvg aggregation with variable participation"""
        total_samples = sum([w['num_samples'] for w in client_weights])
        aggregated_weights = OrderedDict()
        
        for key in self.global_weights.keys():
            aggregated_weights[key] = torch.zeros_like(self.global_weights[key])
            for w in client_weights:
                aggregated_weights[key] += w['weights'][key] * (w['num_samples'] / total_samples)
        
        self.global_weights = aggregated_weights
        self.model.load_state_dict(self.global_weights)
        
        # Update all clients with new global model
        for client in self.clients:
            client.model.load_state_dict(self.global_weights)
    
    def run_experiment(self):
        for round in range(1, self.args.rounds + 1):
            # Track solo performance
            solo_accs = []
            for client in self.clients:
                client.model.load_state_dict(self.global_weights)
                solo_acc = client.evaluate(self.test_loader)
                solo_accs.append(solo_acc)
            avg_solo = np.mean(solo_accs)
            
            # Federated training with different participation rates
            fed_accs = {}
            for rate in [0.25, 0.5, 0.75]:
                num_participants = max(1, int(rate * self.args.num_clients))
                participants = np.random.choice(self.clients, num_participants, replace=False)
                
                client_weights = []
                for client in participants:
                    weights = client.train(self.global_weights)
                    client_weights.append(weights)
                
                self.aggregate(client_weights, rate)
                fed_acc = self.evaluate()
                fed_accs[rate] = fed_acc
            
            # Record results at specified rounds
            if round in [1, 2, 5, 10, 15, 20]:
                self.results['rounds'].append(round)
                self.results['solo_acc'].append(avg_solo)
                self.results['fed_acc_25'].append(fed_accs[0.25])
                self.results['fed_acc_50'].append(fed_accs[0.5])
                self.results['fed_acc_75'].append(fed_accs[0.75])
                
                print(f"Round {round}: Solo={avg_solo:.2f}%, 25%={fed_accs[0.25]:.2f}%, "
                      f"50%={fed_accs[0.5]:.2f}%, 75%={fed_accs[0.75]:.2f}%")
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total