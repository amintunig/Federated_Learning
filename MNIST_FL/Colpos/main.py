# main.py
import os
import tensorflow as tf
from model import build_resnet50
from server import FederatedServer
from client import FederatedClient
from utils import load_data, evaluate_model, visualize_results

# Define data paths and parameters
img_size = 512
batch_size = 32
train_path = "D:/New_FL_Project/COMPUTATION_TIME_ASSIGN/TRAIN_NEW_IMAGE"
test_path = "D:/New_FL_Project/COMPUTATION_TIME_ASSIGN/VAL_NEW_IMAGE"
num_clients = 5
federated_rounds = 5
epochs_per_round = 2

# Load the datasets
train_generator, val_generator, test_generator, num_classes = load_data(train_path, test_path, img_size, batch_size)
input_shape = (img_size, img_size, 3)

# ==========================
# 🚀 Train ResNet-50 (Centralized)
# ==========================
centralized_model = build_resnet50(input_shape, num_classes)
centralized_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])

print("Training ResNet-50 (Centralized)...")
history_resnet = centralized_model.fit(train_generator,
                                       epochs=10,
                                       validation_data=val_generator)

metrics_resnet = evaluate_model(centralized_model, test_generator)
print("\n📊 Centralized ResNet-50 Metrics:")
print(f"Accuracy: {metrics_resnet[0]:.4f}, Precision: {metrics_resnet[1]:.4f}, Recall: {metrics_resnet[2]:.4f}, F1 Score: {metrics_resnet[3]:.4f}")

# ==========================
# 🚀 Federated Learning Simulation
# ==========================
print("\nTraining Federated ResNet-50...")
global_model = build_resnet50(input_shape, num_classes)
server = FederatedServer(global_model)
clients = [
    FederatedClient(i, train_path, img_size, batch_size, num_classes)
    for i in range(num_clients)
]

for round_num in range(federated_rounds):
    print(f"\nRound {round_num + 1}/{federated_rounds}...")
    client_models = []
    for client in clients:
        print(f"  Client {client.client_id + 1} training...")
        local_model = tf.keras.models.clone_model(server.get_global_model())
        local_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss="categorical_crossentropy",
                            metrics=["accuracy"])
        trained_model = client.train_model(local_model, epochs=epochs_per_round)
        client_models.append(trained_model)

    print("  Server aggregating weights...")
    server.aggregate_weights(client_models)

federated_model = server.get_global_model()
metrics_federated = evaluate_model(federated_model, test_generator)

print("\n📊 Federated ResNet-50 Metrics:")
print(f"Accuracy: {metrics_federated[0]:.4f}, Precision: {metrics_federated[1]:.4f}, Recall: {metrics_federated[2]:.4f}, F1 Score: {metrics_federated[3]:.4f}")

# ==========================
# 🚀 Visualization
# ==========================
visualize_results(metrics_resnet, metrics_federated)