"""
Configuration file for Centralized Training
Contains all hyperparameters and settings for centralized model training
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import torch


@dataclass
class CentralizedConfig:
    """Configuration class for centralized training"""
    
    # Data Configuration
    data_path: str = "D:/Ascl_Mimic_Data/CC_Kaggle_Datasets"
    dataset_name: str = "SiPaKMeD"
    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3
    num_classes: int = 5  # koilocytoplastic, metaplastic, dyskaeratatic, parabase, superficial
    class_names: List[str] = field(default_factory=lambda: [
        "koilocytoplastic", "metaplastic", "dyskaeratatic", "parabase", "superficial"
    ])
    total_samples: int = 4049
    
    # Data preprocessing
    normalize_pixels: bool = True
    augment_data: bool = True
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Data augmentation parameters
    rotation_range: float = 20.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    
    # Model Architecture
    model_type: str = "cnn"  # Options: "cnn", "resnet", "vgg", "custom"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    
    # CNN Architecture (if model_type == "cnn")
    conv_layers: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    conv_kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    dropout_rate: float = 0.5
    dense_layers: List[int] = field(default_factory=lambda: [512, 256])
    
    # Training Configuration
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # Options: "adam", "sgd", "rmsprop"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_schedule_type: str = "reduce_on_plateau"  # Options: "reduce_on_plateau", "exponential", "cosine"
    lr_reduction_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-7
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 15
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.001
    
    # Model checkpointing
    save_best_model: bool = True
    checkpoint_monitor: str = "val_accuracy"
    save_weights_only: bool = False
    
    # Regularization
    use_batch_normalization: bool = True
    use_dropout: bool = True
    l2_regularization: float = 0.001
    grad_clip: float = 1.0
    
    # GPU Configuration
    use_mixed_precision: bool = False
    gpu_memory_growth: bool = True
    
    # Class weights (optional, for imbalanced datasets)
    class_weights: Optional[List[float]] = None
    
    # Paths
    model_save_path: str = "models/centralized_model.pth"
    checkpoint_path: str = "checkpoints/centralized/"
    log_dir: str = "logs/centralized/"
    
    def __post_init__(self):
        """Post-initialization to create directories and validate settings"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate configuration
        assert self.num_classes == len(self.class_names), \
            f"Number of classes ({self.num_classes}) doesn't match class names ({len(self.class_names)})"
        
        assert 0 < self.validation_split < 1, "Validation split must be between 0 and 1"
        assert 0 < self.test_split < 1, "Test split must be between 0 and 1"
        assert (self.validation_split + self.test_split) < 1, \
            "Sum of validation and test splits must be less than 1"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'data_path': self.data_path,
            'dataset_name': self.dataset_name,
            'image_size': self.image_size,
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_samples': self.total_samples,
            'normalize_pixels': self.normalize_pixels,
            'augment_data': self.augment_data,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'patience': self.patience,
            'grad_clip': self.grad_clip,
            'class_weights': self.class_weights,
            'model_type': self.model_type,
            'conv_layers': self.conv_layers,
            'dense_layers': self.dense_layers,
            'dropout_rate': self.dropout_rate
        }