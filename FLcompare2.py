import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define the residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First Conv
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second Conv
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match shortcut dimensions
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Create the High ResNet-50 model
def create_high_resnet_50():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, (7, 7), padding='same', strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(3):
        x = residual_block(x, 64)

    x = layers.Conv2D(128, (1, 1), strides=2, padding='same')(x)
    for _ in range(4):
        x = residual_block(x, 128)

    x = layers.Conv2D(256, (1, 1), strides=2, padding='same')(x)
    for _ in range(6):
        x = residual_block(x, 256)

    x = layers.Conv2D(512, (1, 1), strides=2, padding='same')(x)
    for _ in range(3):
        x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = tf.image.resize(x_train, (256, 256)).numpy()
    x_test = tf.image.resize(x_test, (256, 256)).numpy()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    return (x_train, y_train), (x_test, y_test)

# Evaluate model
def evaluate_model(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Train and evaluate
(x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
model = create_high_resnet_50()

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
