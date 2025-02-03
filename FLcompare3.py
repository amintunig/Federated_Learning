import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# 🚀 ResNet-50 Implementation (From Scratch)
# ==========================

def residual_block(x, filters, downsample=False):
    """Defines a residual block with optional downsampling."""
    shortcut = x
    stride = 2 if downsample else 1

    # Main Path
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Shortcut Path
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Merge Paths
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet50(input_shape, num_classes):
    """Builds a ResNet-50 model from scratch."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Residual Blocks
    for filters, downsample in zip([64, 128, 256, 512], [False, True, True, True]):
        for _ in range(3):
            x = residual_block(x, filters, downsample=downsample)
            downsample = False  # Only downsample on first block of each stage

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


# ==========================
# 🚀 Load and Preprocess Data
# ==========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             horizontal_flip=True, brightness_range=[0.8, 1.2])
datagen.fit(x_train)

# ==========================
# 🚀 Train ResNet-50 (Centralized)
# ==========================
input_shape = x_train.shape[1:]
resnet_model = build_resnet50(input_shape, num_classes)

resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss="categorical_crossentropy", metrics=["accuracy"])

print("Training ResNet-50 (Centralized)...")
history_resnet = resnet_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                                  epochs=20, validation_data=(x_test, y_test))


# ==========================
# 🚀 Federated Learning Simulation
# ==========================

def federated_learning_simulation(model, x_train, y_train, num_clients=5, rounds=10, epochs_per_round=5):
    """Simulates Federated Learning with weighted FedAvg."""
    client_sizes = np.random.randint(5000, 10000, num_clients)
    client_indices = np.cumsum([0] + list(client_sizes))
    
    global_model = model

    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}...")

        client_models = []
        for i in range(num_clients):
            start, end = client_indices[i], client_indices[i + 1]
            x_client, y_client = x_train[start:end], y_train[start:end]

            # Clone and train client model
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                 loss="categorical_crossentropy",
                                 metrics=["accuracy"])
            client_model.fit(x_client, y_client, epochs=epochs_per_round, batch_size=32, verbose=0)
            client_models.append(client_model)

        # Weighted FedAvg
        total_samples = sum(client_sizes)
        new_weights = [
            np.sum([client_models[i].get_weights()[layer] * (client_sizes[i] / total_samples)
                    for i in range(num_clients)], axis=0)
            for layer in range(len(global_model.get_weights()))
        ]

        global_model.set_weights(new_weights)

    return global_model


print("Training Federated ResNet-50...")
federated_model = build_resnet50(input_shape, num_classes)
federated_model = federated_learning_simulation(federated_model, x_train, y_train)


# ==========================
# 🚀 Model Evaluation
# ==========================
def evaluate_model(model, x_test, y_test):
    """Evaluates model and returns accuracy, precision, recall, and F1-score."""
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1


metrics_resnet = evaluate_model(resnet_model, x_test, y_test)
metrics_federated = evaluate_model(federated_model, x_test, y_test)

print("\n📊 Centralized ResNet-50 Metrics:")
print(f"Accuracy: {metrics_resnet[0]:.4f}, Precision: {metrics_resnet[1]:.4f}, Recall: {metrics_resnet[2]:.4f}, F1 Score: {metrics_resnet[3]:.4f}")

print("\n📊 Federated ResNet-50 Metrics:")
print(f"Accuracy: {metrics_federated[0]:.4f}, Precision: {metrics_federated[1]:.4f}, Recall: {metrics_federated[2]:.4f}, F1 Score: {metrics_federated[3]:.4f}")


# ==========================
# 🚀 Visualization of Results
# ==========================
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, metrics_resnet, width, label="ResNet-50")
ax.bar(x + width / 2, metrics_federated, width, label="Federated ResNet-50")

ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
