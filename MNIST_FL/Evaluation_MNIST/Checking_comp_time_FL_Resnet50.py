import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# 🚀 Load Local Dataset
# ==========================
img_size = 512  
batch_size = 32
train_path = "D:/New_Project/COMPUTATION_TIME_ASIIGN/TRAIN_NEW_IMAGE"
test_path = "D:/New_Project/COMPUTATION_TIME_ASIIGN/VAL_NEW_IMAGE"

datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

test_generator = datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)
input_shape = (img_size, img_size, 3)

# ==========================
# 🚀 ResNet-50 Model
# ==========================
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet50(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    for filters, downsample in zip([64, 128, 256, 512], [False, True, True, True]):
        for _ in range(3):
            x = residual_block(x, filters, downsample=downsample)
            downsample = False

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# ==========================
# 🚀 Train ResNet-50 (Centralized)
# ==========================
resnet_model = build_resnet50(input_shape, num_classes)

resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

print("Training ResNet-50 (Centralized)...")
history_resnet = resnet_model.fit(train_generator,
                        epochs=10,
                        validation_data=val_generator)

# ==========================
# 🚀 Federated Learning Simulation
# ==========================
def federated_learning_simulation(model, train_data, num_clients=5, rounds=5, epochs_per_round=2):
    """ Simulates Federated Learning by training on client data and averaging weights. """
    
    # Split training images into clients
    filepaths = np.array(train_data.filepaths)
    labels = np.array(train_data.classes)
    client_data = np.array_split(filepaths, num_clients)
    client_labels = np.array_split(labels, num_clients)
    
    global_model = model

    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}...")

        client_models = []
        for i in range(num_clients):
            # Create new data generators for each client
            client_datagen = ImageDataGenerator(rescale=1.0/255.0)
            client_generator = client_datagen.flow_from_directory(
                directory=train_path,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode="categorical"
            )

            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                loss="categorical_crossentropy",
                                metrics=["accuracy"])
            client_model.fit(client_generator, epochs=epochs_per_round, verbose=0)
            client_models.append(client_model)

        # Federated averaging of model weights
        new_weights = [
            np.mean([client.get_weights()[layer] for client in client_models], axis=0)
            for layer in range(len(global_model.get_weights()))
        ]

        global_model.set_weights(new_weights)

    return global_model

print("Training Federated ResNet-50...")
federated_model = build_resnet50(input_shape, num_classes)
federated_model = federated_learning_simulation(federated_model, train_generator)

# ==========================
# 🚀 Model Evaluation
# ==========================
def evaluate_model(model, test_data):
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_true = test_data.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1

metrics_resnet = evaluate_model(resnet_model, test_generator)
metrics_federated = evaluate_model(federated_model, test_generator)

print("\n📊 Centralized ResNet-50 Metrics:")
print(f"Accuracy: {metrics_resnet[0]:.4f}, Precision: {metrics_resnet[1]:.4f}, Recall: {metrics_resnet[2]:.4f}, F1 Score: {metrics_resnet[3]:.4f}")

print("\n📊 Federated ResNet-50 Metrics:")
print(f"Accuracy: {metrics_federated[0]:.4f}, Precision: {metrics_federated[1]:.4f}, Recall: {metrics_federated[2]:.4f}, F1 Score: {metrics_federated[3]:.4f}")

# ==========================
# 🚀 Visualization
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
