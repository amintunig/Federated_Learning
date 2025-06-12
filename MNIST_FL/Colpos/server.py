# server.py
import numpy as np
import tensorflow as tf

class FederatedServer:
    def __init__(self, global_model):
        """Initializes the Federated Server with a global model."""
        self.global_model = global_model

    def aggregate_weights(self, client_models):
        """Averages the weights from the client models."""
        global_weights = self.global_model.get_weights()
        new_weights = [
            np.mean([client.get_weights()[layer] for client in client_models], axis=0)
            for layer in range(len(global_weights))
        ]
        self.global_model.set_weights(new_weights)

    def get_global_model(self):
        """Returns the current global model."""
        return self.global_model