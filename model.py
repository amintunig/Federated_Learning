import tensorflow as tf
from tensorflow.keras import layers, models

def create_resnet50(input_shape=(512, 512, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Layer
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Define identity and convolutional blocks
    def identity_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters[0], (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters[1], (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters[2], (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    def convolutional_block(x, filters):
        shortcut = layers.Conv2D(filters[2], (1, 1), strides=(2, 2))(x)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters[0], (1, 1), strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters[1], (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters[2], (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Build ResNet50 architecture
    x = convolutional_block(x, [64, 64, 256])
    for _ in range(2):
        x = identity_block(x, [64, 64, 256])

    x = convolutional_block(x, [128, 128, 512])
    for _ in range(3):
        x = identity_block(x, [128, 128, 512])

    x = convolutional_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    x = convolutional_block(x, [512, 512, 2048])
    for _ in range(2):
        x = identity_block(x, [512, 512, 2048])

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
