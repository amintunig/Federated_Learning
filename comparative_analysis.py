#filename=comparative_analysis.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from high_resnet50 import build_high_resnet50
from federated_high_resnet50 import federated_learning

# Load dataset paths and parameters
train_path = "D:/Bari_2024_Project/FLdatasets/Dtrain"
test_path = "D:/Bari_2024_Project/FLdatasets/Dtest"
img_size = 64  # Adjust as needed
batch_size = 4
num_classes = 4

# Data Augmentation
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Train and Validation Generators
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="sparse",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="sparse",
    subset="validation"
)

test_generator = datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

# Train Standard High ResNet-50 Model
standard_model = build_high_resnet50((img_size, img_size, 3), num_classes)
standard_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
print("\n🚀 Training Standard High ResNet-50...")
standard_model.fit(train_generator, epochs=10, validation_data=val_generator)

# Train Federated High ResNet-50 Model
federated_model = build_high_resnet50((img_size, img_size, 3), num_classes)
federated_model = federated_learning(federated_model, train_generator)

# Evaluate Models
def evaluate_model(model, test_data):
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_true = test_data.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return accuracy, precision, recall, f1

metrics_standard = evaluate_model(standard_model, test_generator)
metrics_federated = evaluate_model(federated_model, test_generator)

print("\n📊 Standard High ResNet-50 Metrics:")
print(f"✅ Accuracy: {metrics_standard[0]:.4f}, Precision: {metrics_standard[1]:.4f}, Recall: {metrics_standard[2]:.4f}, F1 Score: {metrics_standard[3]:.4f}")

print("\n📊 Federated High ResNet-50 Metrics:")
print(f"✅ Accuracy: {metrics_federated[0]:.4f}, Precision: {metrics_federated[1]:.4f}, Recall: {metrics_federated[2]:.4f}, F1 Score: {metrics_federated[3]:.4f}")