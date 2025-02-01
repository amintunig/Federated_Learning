#filename=federated_high_resnet50.py
import numpy as np
import tensorflow as tf
from high_resnet50 import build_high_resnet50

def federated_learning(model, train_data, num_clients=5, rounds=5, epochs_per_round=2):
    """ Simulates Federated Learning by training on client data and averaging weights. """
    
    # Split training data into clients
    client_data = np.array_split(train_data, num_clients)
    
    global_model = model

    for round_num in range(rounds):
        print(f"\n🔄 Round {round_num + 1}/{rounds}...")
        client_models = []
        
        for i, client in enumerate(client_data):
            print(f"  📡 Training Client {i + 1}/{num_clients}...")
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                 loss="sparse_categorical_crossentropy",
                                 metrics=["accuracy"])
            client_model.fit(client, epochs=epochs_per_round, verbose=0)
            client_models.append(client_model)

        # Federated averaging of model weights
        new_weights = [
            np.mean([client.get_weights()[layer] for client in client_models], axis=0)
            for layer in range(len(global_model.get_weights()))
        ]
        global_model.set_weights(new_weights)

    return global_model