from train_centralized import create_dataloaders, create_model, CentralizedTrainer
from config_centralized import CentralizedConfig
import torch
from train_centralized import main
def main():
    # Load configuration
    config = CentralizedConfig()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = CentralizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config
    )
    
    # Train the model
    history = trainer.train()
    
    # Test the model
    test_results = trainer.test()
    
    # Plot training history
    trainer.plot_training_history()

if __name__ == "__main__":
    main()