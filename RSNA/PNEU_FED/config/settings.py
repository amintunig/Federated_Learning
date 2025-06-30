import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "D:/Ascl_Mimic_Data/RSNA")
    TRAIN_IMG_DIR: str = os.path.join(DATA_DIR, "Training/Images")
    TEST_IMG_DIR: str = os.path.join(DATA_DIR, "Test/Images")
    META_PATH: str = os.path.join(DATA_DIR, "stage2_train_metadata.csv")
    
    # Model settings
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_CLIENTS: int = 3
    FL_ROUNDS: int = 10
    
    # Federated learning
    SERVER_ADDRESS: str = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")
    CLIENT_TIMEOUT: int = 600
    
    # Training
    LEARNING_RATE: float = 0.001
    DROPOUT_RATE: float = 0.5
    
    # Data partitioning
    PARTITION_SCENARIO: int = 1  # 1=IID, 2=Non-IID
    TEST_SIZE: float = 0.2
    
    # Logging
    LOG_DIR: str = "logs"
    MODEL_SAVE_PATH: str = "saved_models"
    
    def __post_init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)

# Initialize settings
settings = Settings()