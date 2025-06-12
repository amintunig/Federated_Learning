# client.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class FederatedClient:
    def __init__(self, client_id, train_path, img_size, batch_size, num_classes):
        """Initializes a Federated Client with its local data."""
        self.client_id = client_id
        self.train_path = train_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_generator = self._load_client_data()

    def _load_client_data(self):
        """Loads the local training data for the client."""
        client_datagen = ImageDataGenerator(rescale=1.0/255.0)
        client_generator = client_datagen.flow_from_directory(
            directory=self.train_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True # Important to shuffle local data
        )
        # Simulate client-specific data by taking a subset (this is the "flower" part)
        # In a real scenario, each client would have its own distinct data.
        num_samples = len(client_generator.filepaths)
        indices = np.random.choice(num_samples, size=num_samples // 5, replace=False) # Simulate smaller local dataset
        filepaths = np.array(client_generator.filepaths)[indices]
        labels = np.array(client_generator.labels)[indices]

        # Create a new ImageDataGenerator and flow from the subset
        subset_datagen = ImageDataGenerator(rescale=1.0/255.0)
        subset_generator = subset_datagen.flow(
            np.array([tf.keras.utils.load_img(fp, target_size=(self.img_size, self.img_size)) for fp in filepaths]),
            tf.keras.utils.to_categorical(labels, num_classes=self.num_classes),
            batch_size=self.batch_size,
            shuffle=True
        )
        return subset_generator

    def train_model(self, model, epochs=1):
        """Trains the provided model on the client's local data."""
        model.fit(self.train_generator, epochs=epochs, verbose=0)
        return model