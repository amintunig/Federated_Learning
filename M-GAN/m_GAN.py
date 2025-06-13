import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path

## 1. Dataset Preparation
def load_sipakmed_dataset(data_dir, img_size=(240, 240)):
    """
    Load SIPaKMeD dataset with 4049 images
    Returns images and labels for 5 classes
    """
    classes = ['im_Koilocytotic', 'im_Metaplastic', 'im_Dyskeratotic', 'im_Parabasal', 'im_Superficial-Intermediate']
    images = []
    labels = []
    failed_images = []  # Initialize failed_images list
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        # Check if the directory exists
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found {class_dir}")
            continue
        print(f"Loading images from: {class_dir}")
        # Get list of images from directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_name in tqdm(image_files, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                # Check if image was loaded properly
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    failed_images.append(img_path)
                    continue

                # Convert BGR to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                # Normalize
                img = img.astype(np.float32) / 255.0

                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                failed_images.append(img_path)
                continue
    
    print(f"\nSuccessfully loaded {len(images)} images")
    print(f"Failed to load {len(failed_images)} images")
    if failed_images:
        print("\nFailed images:")
        for img_path in failed_images[:10]:  # Show only first 10 failed images
            print(img_path)
    
    return np.array(images), np.array(labels)

# Load dataset (replace with your SIPaKMeD dataset path)
data_dir = "D:/Ascl_Mimic_Data/CC_Kaggle_Datasets"
directories = [
    "im_Koilocytotic/im_Koilocytotic",
    "im_Metaplastic/im_Metaplastic", 
    "im_Dyskeratotic/im_Dyskeratotic",
    "im_Parabasal/im_Parabasal",
    "im_Superficial-Intermediate/im_Superficial-Intermediate"
]
images, labels = load_sipakmed_dataset(data_dir, directories)
for dir_name in directories:
    dir_path = Path(data_dir) / dir_name
    print(f"\nChecking directory: {dir_path}")
    print(f"Directory exists: {dir_path.exists()}")
    
    if dir_path.exists():
        files = list(dir_path.glob("*"))
        print(f"Total files: {len(files)}")
        
        # Check for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in files if f.suffix.lower() in image_extensions]
        print(f"Image files: {len(image_files)}")
        
        if image_files:
            print(f"Sample files: {[f.name for f in image_files[:3]]}")

# Split dataset (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

## 2. Preprocessing Phase - Contrast Maximization
def contrast_maximization(images):
    """
    Implement contrast maximization using Pixel Expansion across Threshold
    """
    enhanced_images = []
    for img in images:
        # Ensure image is in correct format
        if img.max() <= 1.0:  # If normalized
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img.astype(np.uint8)
            
        # Convert to grayscale
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Group pixels into 3 groups
        group1 = np.where((gray >= 0) & (gray <= 99), gray, 0)
        group2 = np.where((gray >= 100) & (gray <= 169), gray, 0)
        group3 = np.where((gray >= 170) & (gray <= 255), gray, 0)
        
        # Enhance G3 pixels
        delta = 20
        enhanced_g3 = np.where(group3 > 0, np.minimum(group3 + delta, 255), 0)
        
        # Combine groups
        enhanced_gray = group1 + group2 + enhanced_g3
        enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
        
        # Convert back to RGB and normalize
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        enhanced_rgb = enhanced_rgb.astype(np.float32) / 255.0
        enhanced_images.append(enhanced_rgb)
    
    return np.array(enhanced_images)

# Apply contrast maximization
X_train_enhanced = contrast_maximization(X_train)
X_test_enhanced = contrast_maximization(X_test)

## 3. Data Augmentation Phase - Multi-modal GAN (m-GAN)
class mGAN:
    def __init__(self, img_shape=(240, 240, 3)):
        self.img_shape = img_shape
        
        # Build generator and discriminators
        self.generator = self.build_generator()
        self.global_discriminator = self.build_global_discriminator()
        self.local_discriminator = self.build_local_discriminator()
        
        # Compile discriminators
        self.global_discriminator.compile(
            optimizer=optimizers.Adam(0.0002, 0.5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        self.local_discriminator.compile(
            optimizer=optimizers.Adam(0.0002, 0.5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Combined model
        self.combined = self.build_combined()
        
    def build_generator(self):
        """Generator model for m-GAN"""
        inputs = layers.Input(shape=self.img_shape)
        
        # Encoder
        x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Bottleneck
        x = layers.Conv2D(512, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Decoder
        x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)
        
        return models.Model(inputs, outputs, name="Generator")
    
    def build_global_discriminator(self):
        """Global discriminator for entire image"""
        inputs = layers.Input(shape=self.img_shape)
        
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        return models.Model(inputs, outputs, name="Global_Discriminator")
    
    def build_local_discriminator(self):
        """Local discriminator for patches"""
        inputs = layers.Input(shape=(120, 120, 3))  # Half the size
        
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        return models.Model(inputs, outputs, name="Local_Discriminator")
    
    def build_combined(self):
        """Combined model with generator and discriminators"""
        # Make discriminators non-trainable for generator training
        self.global_discriminator.trainable = False
        self.local_discriminator.trainable = False
        
        # Inputs
        img_input = layers.Input(shape=self.img_shape)
        
        # Generate image
        gen_img = self.generator(img_input)
        
        # Discriminator outputs
        global_valid = self.global_discriminator(gen_img)
        local_valid = self.local_discriminator(gen_img[:, :120, :120, :])  # Top-left patch
        
        # Combined model
        combined = models.Model(
            inputs=img_input,
            outputs=[gen_img, global_valid, local_valid],
            name="Combined"
        )
        
        # Compile with custom loss weights
        combined.compile(
            optimizer=optimizers.Adam(0.0002, 0.5),
            loss=["mse", "binary_crossentropy", "binary_crossentropy"],
            loss_weights=[100, 1, 1]  # Higher weight for reconstruction loss
        )
        
        return combined
    
    def train(self, X_train, epochs=50, batch_size=16):
        """Train m-GAN model"""
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Convert images to tanh range [-1, 1] for generator
            imgs_tanh = (imgs * 2.0) - 1.0
            
            # ---------------------
            #  Train Discriminators
            # ---------------------
            
            # Generate augmented images
            gen_imgs = self.generator.predict(imgs_tanh, verbose=0)
            
            # Convert back to [0, 1] range for discriminators
            gen_imgs_norm = (gen_imgs + 1.0) / 2.0
            
            # Train global discriminator
            d_global_loss_real = self.global_discriminator.train_on_batch(imgs, valid)
            d_global_loss_fake = self.global_discriminator.train_on_batch(gen_imgs_norm, fake)
            d_global_loss = 0.5 * np.add(d_global_loss_real, d_global_loss_fake)
            
            # Train local discriminator
            d_local_loss_real = self.local_discriminator.train_on_batch(imgs[:, :120, :120, :], valid)
            d_local_loss_fake = self.local_discriminator.train_on_batch(gen_imgs_norm[:, :120, :120, :], fake)
            d_local_loss = 0.5 * np.add(d_local_loss_real, d_local_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Train generator
            g_loss = self.combined.train_on_batch(imgs_tanh, [imgs_tanh, valid, valid])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [D global loss: {d_global_loss[0]:.4f}, acc: {d_global_loss[1]*100:.1f}%] "
                      f"[D local loss: {d_local_loss[0]:.4f}, acc: {d_local_loss[1]*100:.1f}%] "
                      f"[G loss: {g_loss[0]:.4f}]")
        
        return self.generator

# Initialize and train m-GAN
print("Training m-GAN...")
m_gan = mGAN()
generator = m_gan.train(X_train_enhanced, epochs=50, batch_size=16)

# Generate augmented images
print("Generating augmented images...")
augmented_images = generator.predict((X_train_enhanced * 2.0) - 1.0)
augmented_images = (augmented_images + 1.0) / 2.0  # Convert back to [0, 1]
X_train_augmented = np.concatenate([X_train_enhanced, augmented_images])
y_train_augmented = np.concatenate([y_train, y_train])

print(f"Augmented training set size: {X_train_augmented.shape[0]}")

## 4. Segmentation Phase - Seg-UNet
def build_seg_unet(input_shape=(240, 240, 3), num_classes=5):
    """Build Seg-UNet model for segmentation"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (Downsampling)
    # Block 1
    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip2 = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip3 = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 4
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip4 = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Decoder (Upsampling)
    # Block 4
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.concatenate([skip4, x])
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 3
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.concatenate([skip3, x])
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 2
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.concatenate([skip2, x])
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 1
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.concatenate([skip1, x])
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name="Seg_UNet")

# Create Seg-UNet (for demonstration, we'll skip training as it needs segmentation masks)
seg_unet = build_seg_unet()
seg_unet.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Seg-UNet model created (training skipped - requires segmentation masks)")

# For demonstration, we'll use the augmented images directly
segmented_train = X_train_augmented
segmented_test = X_test_enhanced

## 5. Feature Extraction Phase - Denoising Autoencoder
class DenoisingAutoencoder:
    def __init__(self, input_shape=(240, 240, 3)):
        self.input_shape = input_shape
        self.autoencoder, self.encoder = self.build_autoencoder()
        
    def build_autoencoder(self):
        """Build denoising autoencoder"""
        # Encoder
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D(2, padding="same")(x)
        
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
        
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
        encoded = layers.MaxPooling2D(2, padding="same")(x)
        
        # Decoder
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(encoded)
        x = layers.UpSampling2D(2)(x)
        
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.UpSampling2D(2)(x)
        
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.UpSampling2D(2)(x)
        
        decoded = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
        
        # Autoencoder model
        autoencoder = models.Model(inputs, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        
        # Encoder model for feature extraction
        encoder = models.Model(inputs, encoded)
        
        return autoencoder, encoder
    
    def train(self, X_train, epochs=30, batch_size=32):
        """Train the autoencoder"""
        # Add noise for denoising
        noise_factor = 0.3
        X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
        X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
        
        # Train autoencoder
        history = self.autoencoder.fit(
            X_train_noisy, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1
        )
        
        return history
    
    def extract_features(self, X):
        """Extract features using trained encoder"""
        features = self.encoder.predict(X)
        return features

# Create and train denoising autoencoder
print("Training Denoising Autoencoder...")
dae = DenoisingAutoencoder()
dae_history = dae.train(segmented_train, epochs=30, batch_size=32)

# Extract features
print("Extracting features...")
train_features = dae.extract_features(segmented_train)
test_features = dae.extract_features(segmented_test)

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

## 6. Classification Phase - Dense CapsNet
class DenseCapsNet:
    def __init__(self, input_shape, num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        """Build Dense CapsNet model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolutional layer
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        
        # Tensor capsule layers with dense connectivity
        # Tensor Layer 1
        tensor1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        tensor1 = layers.BatchNormalization()(tensor1)
        
        # Tensor Layer 2 (connected to input and tensor1)
        x2 = layers.concatenate([x, tensor1])
        tensor2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x2)
        tensor2 = layers.BatchNormalization()(tensor2)
        
        # Tensor Layer 3 (connected to input, tensor1, tensor2)
        x3 = layers.concatenate([x, tensor1, tensor2])
        tensor3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x3)
        tensor3 = layers.BatchNormalization()(tensor3)
        
        # Tensor Layer 4 (connected to input, tensor1-3)
        x4 = layers.concatenate([x, tensor1, tensor2, tensor3])
        tensor4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x4)
        tensor4 = layers.BatchNormalization()(tensor4)
        
        # Tensor Layer 5 (connected to input, tensor1-4)
        x5 = layers.concatenate([x, tensor1, tensor2, tensor3, tensor4])
        tensor5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x5)
        tensor5 = layers.BatchNormalization()(tensor5)
        
        # Dense Capsule Layer
        x_caps = layers.concatenate([tensor1, tensor2, tensor3, tensor4, tensor5])
        x_caps = layers.GlobalAveragePooling2D()(x_caps)
        x_caps = layers.Dropout(0.5)(x_caps)
        
        # Classification Layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x_caps)
        
        # Create model
        model = models.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the Dense CapsNet model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        return self.model.evaluate(X_test, y_test, verbose=1)

# Create and train Dense CapsNet
print("Training Dense CapsNet...")
dense_capsnet = DenseCapsNet(input_shape=train_features.shape[1:])
history = dense_capsnet.train(train_features, y_train_augmented, epochs=50, batch_size=32)

# Evaluate on test set
print("Evaluating model...")
test_loss, test_acc = dense_capsnet.evaluate(test_features, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Generate predictions
y_pred = np.argmax(dense_capsnet.model.predict(test_features), axis=1)

# Classification report
class_names = ['Koilocytotic', 'Metaplastic', 'Dyskeratotic', 'Parabasal', 'Superficial-Intermediate']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

## 7. Visualization of Results
def plot_history(history, title="Model Training History"):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()