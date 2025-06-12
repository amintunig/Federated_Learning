import tensorflow as tf
import flwr as fl
import numpy as np

# =======================
# 📦 Load and Preprocess Data
# =======================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Squeeze labels to shape (n,)
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# =======================
# 🧠 Build Model
# =======================
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    weights=None,
    classes=10
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =======================
# 🌸 Flower Client
# =======================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(accuracy)}

# =======================
# 🚀 Start Client
# =======================
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model, x_train, y_train, x_test, y_test)
    )
