# utils.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def load_data(train_path, test_path, img_size, batch_size):
    """Loads and preprocesses the image datasets."""
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

    return train_generator, val_generator, test_generator, len(train_generator.class_indices)

def evaluate_model(model, test_data):
    """Evaluates the given model on the test data."""
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_true = test_data.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1

def visualize_results(metrics_centralized, metrics_federated):
    """Visualizes the comparison of centralized and federated learning metrics."""
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, metrics_centralized, width, label="Centralized Model")
    ax.bar(x + width / 2, metrics_federated, width, label="Federated Model")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()