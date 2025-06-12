"""
Centralized Training Script for Cervical Cancer Classification
Handles the complete training pipeline for centralized learning
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import time
import logging
import os
import json
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config_centralized import CentralizedConfig
from evaluate_centralized import MetricsCalculator, count_parameters


class CervicalCancerCNN(nn.Module):
    def __init__(self, config: CentralizedConfig):
        super(CervicalCancerCNN, self).__init__()
        self.config = config
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = config.num_channels
        
        for out_channels in config.conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=config.conv_kernel_size, 
                             padding='same'),
                    nn.BatchNorm2d(out_channels) if config.use_batch_normalization else nn.Identity(),
                    nn.ReLU(),
                    nn.MaxPool2d(config.pool_size)
                )
            )
            in_channels = out_channels
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(2, config.num_channels, *config.image_size)  # Use batch size 2
            for layer in self.conv_layers:
                dummy_input = layer(dummy_input)
            dummy_input = self.adaptive_pool(dummy_input)
            flattened_size = dummy_input.numel() // 2  # Divide by batch size
        
        # Dense layers with batch norm only if batch size > 1
        self.dense_layers = nn.ModuleList()
        in_features = flattened_size
        
        for out_features in config.dense_layers:
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features) if (config.use_batch_normalization and 
                                                   config.batch_size > 1) else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate) if config.use_dropout else nn.Identity()
                )
            )
            in_features = out_features
        
        # Output layer
        self.output_layer = nn.Linear(in_features, config.num_classes)
    
    def forward(self, x):
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x


def create_datasets(config: CentralizedConfig):
    """Create train, validation, and test datasets"""
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=config.data_path, transform=transform)
    
    # Split into train, validation, and test
    val_size = int(config.validation_split * len(full_dataset))
    test_size = int(config.test_split * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size
    
    # Ensure we have enough samples
    if train_size < 2 or val_size < 2 or test_size < 2:
        raise ValueError("Not enough samples in dataset. Need at least 2 samples per split.")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(config: CentralizedConfig):
    """Create data loaders for training, validation, and testing"""
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    
    # Create data loaders with drop_last=True to avoid single-sample batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(config.batch_size, len(train_dataset)), 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(config.batch_size, len(test_dataset)),
        shuffle=False,
        drop_last=False  # Can be False for testing
    )
    
    return train_loader, val_loader, test_loader


def create_model(config: CentralizedConfig):
    """Create model based on configuration"""
    if config.model_type == "cnn":
        return CervicalCancerCNN(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


class CentralizedTrainer:
    """Centralized training for cervical cancer classification"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 config: CentralizedConfig):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function with class weights if provided
        self.criterion = self._setup_criterion()
        
        # Setup metrics
        self.metrics_calculator = MetricsCalculator(
            num_classes=config.num_classes,
            class_names=config.class_names
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if not self.config.use_lr_scheduler:
            return None
        
        if self.config.lr_schedule_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=self.config.lr_patience,
                factor=self.config.lr_reduction_factor,
                min_lr=self.config.min_lr
            )
        elif self.config.lr_schedule_type == 'exponential':
            return StepLR(self.optimizer, step_size=10, gamma=0.9)
        elif self.config.lr_schedule_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        else:
            return None
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function"""
        if self.config.class_weights is not None:
            weights = torch.FloatTensor(self.config.class_weights).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.log_dir, 'training.log')
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct_predictions += predicted.eq(targets).sum().item()
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Loss: {loss.item():.4f}, '
                    f'Acc: {100. * correct_predictions / total_samples:.2f}%'
                )
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Update metrics
                self.metrics_calculator.update(predicted, targets, probabilities)
        
        val_loss = running_loss / len(self.val_loader)
        metrics = self.metrics_calculator.compute_metrics()
        
        return val_loss, metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Complete training loop"""
        self.logger.info("Starting centralized training...")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info("-" * 50)
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Log epoch results
            self.logger.info(
                f'Epoch {epoch+1:3d}/{self.config.epochs} ({epoch_time:.1f}s) | '
                f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} Acc: {val_metrics["accuracy"]:.4f} | '
                f'LR: {current_lr:.6f}'
            )
            
            # Save best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                is_best = True
                self.save_checkpoint('best_model.pth', epoch, is_best=True)
                self.logger.info(f"New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if self.config.use_early_stopping and self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
                break
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final training history
        self.save_training_history()
        
        return self.history
    
    def test(self) -> Dict[str, float]:
        """Test the model on test set"""
        self.logger.info("Testing model...")
        
        # Load best model if it exists
        best_model_path = os.path.join(self.config.checkpoint_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.logger.info("Loading best model for testing...")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        self.metrics_calculator.reset()
        test_loss = 0.0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                self.metrics_calculator.update(predicted, targets, probabilities)
        
        test_metrics = self.metrics_calculator.compute_metrics()
        test_metrics['loss'] = test_loss / len(self.test_loader)
        
        # Log test results
        self.logger.info("=" * 60)
        self.logger.info("TEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        self.logger.info(f"Test Precision (Macro): {test_metrics['precision_macro']:.4f}")
        self.logger.info(f"Test Recall (Macro): {test_metrics['recall_macro']:.4f}")
        
        # Per-class results
        self.logger.info("\nPer-class Results:")
        for i, class_name in enumerate(self.config.class_names):
            precision = test_metrics.get(f'precision_{class_name}', 0)
            recall = test_metrics.get(f'recall_{class_name}', 0)
            f1 = test_metrics.get(f'f1_{class_name}', 0)
            self.logger.info(f"{class_name:15s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        self.logger.info("=" * 60)
        
        # Save confusion matrix and ROC curves
        save_dir = os.path.join(self.config.log_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_calculator.plot_confusion_matrix(
            os.path.join(save_dir, 'test_confusion_matrix.png')
        )
        
        if self.metrics_calculator.all_probabilities:
            self.metrics_calculator.plot_roc_curves(
                os.path.join(save_dir, 'test_roc_curves.png')
            )
        
        # Save test metrics
        with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, (np.float64, np.float32)):
                    serializable_metrics[k] = float(v)
                elif isinstance(v, (np.int64, np.int32)):
                    serializable_metrics[k] = int(v)
                else:
                    serializable_metrics[k] = v
            json.dump(serializable_metrics, f, indent=2)
        
        # Generate and save classification report
        report = self.metrics_calculator.generate_classification_report()
        with open(os.path.join(save_dir, 'test_classification_report.txt'), 'w') as f:
            f.write(report)
        
        return test_metrics
    
    def save_checkpoint(self, filename: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_path, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'is_best': is_best
        }
        
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            self.logger.info(f"Best model checkpoint saved: {checkpoint_path}")
        else:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return epoch number"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from epoch {epoch + 1}")
        
        return epoch
    
    def save_training_history(self):
        """Save training history to file"""
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        
        # Convert to serializable format
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                       for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        if save_path is None:
            save_path = os.path.join(self.config.log_dir, 'training_plots.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['lr'], color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Epoch time plot
        axes[1, 1].plot(self.history['epoch_times'], color='orange')
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved: {save_path}")


def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = CentralizedConfig()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
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
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()