import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

class Client:
    def __init__(self, model, train_loader, args, client_id=None):
        self.model = model
        self.train_loader = train_loader
        self.args = args
        self.client_id = client_id
        self.optimizer = optim.SGD(self.model.parameters(), 
                                  lr=args.lr, 
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)
        
    def train(self, global_weights=None):
        if global_weights is not None:
            self.model.load_state_dict(global_weights)
        
        self.model.train()
        for epoch in range(self.args.local_epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        return {
            'weights': self.model.state_dict(),
            'num_samples': len(self.train_loader.dataset)
        }
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total