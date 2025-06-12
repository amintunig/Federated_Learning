import tensorflow as tf 
from tensorflow.keras import layers, models 
import numpy as np
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')

#Load cifar10 datasets

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Normalize the datasets
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#one_hot encoding labels

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print(f"Train shape: {x_train.shape}, {y_train.shape}")
print(f"Test shape: {x_test.shape}, {y_test.shape}")

#Spliting the data across the client

def split_data(x, y, num_clients):
    client_data = []
    shard_size = len(x) // num_clients
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_data.append((x[start:end], y[start:end]))
    return client_data

# Split data for 3 clients
num_clients = 3
client_data = split_data(x_train, y_train, num_clients)

#Define the model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32, 3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
#Initialize the global model
global_model = create_model()

#Each local model starts with the same initial weights
initial_weights = global_model.get_weights()

#Federated Learning Training
#number of rounds for federated learning
num_rounds = 10
for round_num in range(num_rounds):
    print(f"Round {round_num + 1} / {num_rounds}")

    local_weights = []

    #training on each clients

    for clients_id in range(num_clients):
        print(f"Training on client {clients_id}")

        #create local model and set global weights
        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        #Get client data
        X, y = client_data[clients_id]

        #Training local model
        local_model.fit(X, y, epochs =1, batch_size = 32, verbose =0)
        #Collect the local model weights
        local_weights.append(local_model.get_weights())

    #Federated Averaging : Aggregated local weights
    average_weights = [np.mean([local_weights[j][i] for j in range(num_clients)], axis=0) for i in range(len(local_weights[0]))]

    #Upadte global model weights
    global_model.set_weights(average_weights)

    #Evaluate the global model
    #loss, accuracy = global_model.evaluate(x_test, y_test, verbose =8)
    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Global model accuracy after round {round_num + 1}: {accuracy:.4f}")

import flwr as fl

class CIFAR10Client(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
    def get_parameters(self):
        return self.model.get_weights()
    def set_parameters(self, parameters):
        return self.model.set_weights(parameters)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        x_train, y_train = self.train_data
        self.model.fit(x_train, y_train, epochs =1, batch_size =32, verbose =0)
        return self.get_parameters(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        x_test, y_test = self.test_data
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose =0)
        return float(loss), len(x_test), {"accuracy" : float(accuracy)}
    
    #start the flower server
    def start_flower_server():
        fl.server.start_server(
        server_address="0.0.0.0:8098",
        config=fl.server.ServerConfig(num_rounds=3),
    )
    
    #Start the flower client
    def start_flower_client(client_id):
        #Get the client specific data
        train_data = client_data[client_id]
        test_data = (x_test, y_test)

        #create model
        model = create_model()

        #Create and start the client
        client = CIFAR10Client(model, train_data, test_data)
        fl.client.start_numpy_client(server_address="0.0.0.0:8098", client=client)
    #Run the Federated Learning

    import flwr as fl

    def start_flower_server():
        strategy = fl.server.strategy.FedAvg()
        config = fl.server.ServerConfig(num_rounds=1)

        #Start the server with the startegy and configuration
        fl.server.start_server(
            server_address= "localhost:8098",
            strategy=strategy,
            config=config,
        )
    
    if __name__ == "__main__":
        start_flower_server()
        
    #start the server directly in the main threa
from multiprocessing import Process
import flwr as fl

def start_flower_client(client_id):
    print(f"Starting client {client_id}")
    model = create_model()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    client = FlowerClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_client(
        server_address="localhost:8098",
        client=client.to_client(),
    )
    if __name__ == "__main__":
        num_clients = 3  # Adjust the number of clients as needed

    # Start clients in separate processes
        client_processes = []
        for client_id in range(num_clients):
            process = Process(target=start_flower_client, args=(client_id,))
            client_processes.append(process)
            process.start()

    # Wait for all clients to finish
        for process in client_processes:
            process.join()
#Evaluate the final global model
#get the final weight from the training global model
global_model_parameters = global_model.get_weights()

#Use the final weights to evaluate
final_model = create_model()
final_model.set_weights(global_model_parameters)
loss, accuracy = final_model.evaluate(x_test, y_test, verbose =0)
print(f"Final model accuracy: {accuracy:.4f}")

#Final More Evaluation
# After federated learning rounds
num_rounds = 5
for round_num in range(num_rounds):
    print(f"Round {round_num + 1}/{num_rounds}")
    
    local_weights = []

    # Train on each client
    for client_id in range(num_clients):
        print(f"Training on client {client_id + 1}")

        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        # Train local model
        X, y = client_data[client_id]
        local_model.fit(X, y, epochs=1, batch_size=32, verbose=0)

        # Collect local weights
        local_weights.append(local_model.get_weights())

    # Federated averaging
    averaged_weights = [np.mean([local_weights[j][i] for j in range(num_clients)], axis=0)
                        for i in range(len(local_weights[0]))]
    global_model.set_weights(averaged_weights)

    # Evaluate global model
    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Global model accuracy after round {round_num + 1}: {accuracy:.4f}")

# Save final weights
global_model_parameters = global_model.get_weights()

# Final evaluation
final_model = create_model()
final_model.set_weights(global_model_parameters)
loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
print(f"Final model accuracy: {accuracy:.4f}")
