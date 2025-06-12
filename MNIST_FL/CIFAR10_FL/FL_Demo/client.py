import flwr as fl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_resnet50

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.model = create_resnet50()

        datagen = ImageDataGenerator(rescale=1.0/255)
        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='categorical'
        )

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.data_generator, epochs=5, steps_per_epoch=len(self.data_generator))
        return self.model.get_weights(), len(self.data_generator), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data_generator)
        return loss, len(self.data_generator), {"accuracy": accuracy}

def start_client(data_dir):
    client = HospitalClient(data_dir)
    fl.client.start_numpy_client("localhost:8080", client=client)
