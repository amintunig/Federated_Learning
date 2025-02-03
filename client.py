import flwr as fl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_resnet50

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.model = create_resnet50()
        
        # Load and preprocess data
        datagen = ImageDataGenerator(rescale=1.0/255)
        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='categorical'
        )

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters: fl.common.NDArrays) -> int:
        self.model.set_weights(parameters)  
        
        # Train the model on local data
        self.model.fit(self.data_generator,
                       epochs=5,
                       steps_per_epoch=len(self.data_generator))
        
        return len(self.data_generator) 

    def evaluate(self) -> float:
        loss_and_accuracy = self.model.evaluate(self.data_generator)
        return loss_and_accuracy[0]   # Return loss

def start_client(data_dir):
    client = HospitalClient(data_dir)
    fl.client.start_numpy_client("localhost:8080", client=client)
