import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add custom residual blocks
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    # Main path
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)

    # Shortcut path
    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)

    # Add shortcut to the main path
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Replace the residual block loop in build_resnet50
def build_resnet50(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Add custom residual blocks
    for filters, downsample in zip([64, 128, 256, 512], [False, True, True, True]):
        for _ in range(3):
            x = residual_block(x, filters, downsample=downsample)
            downsample = False  # Only downsample on the first block of each stage

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build ResNet50 model
input_shape = x_train.shape[1:]
resnet_model = build_resnet50(input_shape, num_classes)
resnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train ResNet50 model
print("Training ResNet50 model...")
history_resnet = resnet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Simulate Federated Learning (split data among clients)
def federated_learning_simulation(model, x_train, y_train, num_clients=5, rounds=3, epochs_per_round=2):
    client_data_size = len(x_train) // num_clients
    global_model = model

    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}...")
        client_models = []

        # Train on each client's data
        for i in range(num_clients):
            start, end = i * client_data_size, (i + 1) * client_data_size
            x_client, y_client = x_train[start:end], y_train[start:end]

            # Clone the global model for the client
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            client_model.fit(x_client, y_client, epochs=epochs_per_round, batch_size=32, verbose=0)
            client_models.append(client_model)

        # Aggregate weights from all clients (FedAvg)
        new_weights = [np.mean([client.get_weights()[layer] for client in client_models], axis=0)
                       for layer in range(len(global_model.get_weights()))]
        global_model.set_weights(new_weights)

    return global_model

# Train ResNet50 with Federated Learning
print("Training Federated ResNet50 model...")
federated_model = build_resnet50(input_shape, num_classes)
federated_model = federated_learning_simulation(federated_model, x_train, y_train)

# Evaluate both models
print("Evaluating models...")
def evaluate_model(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1

metrics_resnet = evaluate_model(resnet_model, x_test, y_test)
metrics_federated = evaluate_model(federated_model, x_test, y_test)

# Plot metrics
labels = ["Accuracy", "Precision", "Recall", "Harmonic Mean"]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(labels))
width = 0.35

ax.bar(x - width / 2, metrics_resnet, width, label="ResNet50")
ax.bar(x + width / 2, metrics_federated, width, label="Federated ResNet50")

ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
