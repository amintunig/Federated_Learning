import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, mnist
import random

# Load the CIFAR-10 dataset
(cifar10_train_images, cifar10_train_labels), (cifar10_test_images, cifar10_test_labels) = cifar10.load_data()

# Load the MNIST dataset
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

# Define class names for CIFAR-10
cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Visualize CIFAR-10 classes and data distribution

# a. Show sample images from CIFAR-10
plt.figure(figsize=(10, 10))
plt.suptitle("CIFAR-10 Sample Images", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(cifar10_train_images[i], cmap=plt.cm.binary)
    plt.xlabel(cifar10_class_names[cifar10_train_labels[i][0]])
    plt.xticks([])
    plt.yticks([])
plt.show()

# b. Show the distribution of CIFAR-10 classes
plt.figure(figsize=(8, 5))
plt.suptitle("CIFAR-10 Class Distribution", fontsize=16)
train_counts = np.bincount(cifar10_train_labels.flatten())
test_counts = np.bincount(cifar10_test_labels.flatten())
plt.bar(cifar10_class_names, train_counts, label='Train')
plt.bar(cifar10_class_names, test_counts, label='Test', alpha=0.7)  # Add test set counts
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# 2. Visualize MNIST classes and data distribution

# a. Show sample images from MNIST
plt.figure(figsize=(10, 10))
plt.suptitle("MNIST Sample Images", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(mnist_train_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Label: {mnist_train_labels[i]}")
    plt.xticks([])
    plt.yticks([])
plt.show()

# b. Show the distribution of MNIST classes
plt.figure(figsize=(8, 5))
plt.suptitle("MNIST Class Distribution", fontsize=16)
train_counts_mnist = np.bincount(mnist_train_labels)
test_counts_mnist = np.bincount(mnist_test_labels) # Get counts for test set
plt.bar(np.arange(10), train_counts_mnist, label='Train')
plt.bar(np.arange(10), test_counts_mnist, label='Test', alpha=0.7) #plot test set counts
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.legend()
plt.xticks(np.arange(10))
plt.show()
