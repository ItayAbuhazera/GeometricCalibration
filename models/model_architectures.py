from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_cifar100_pretrained_efficientnet(input_shape=(32, 32, 3), num_classes=100):
    """
    Pretrained EfficientNetB0 model fine-tuned for CIFAR-100 classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.

    Returns:
        Model: Compiled EfficientNetB0 model for CIFAR-100.
    """
    # Load the EfficientNetB0 model without the top layer (pretrained on ImageNet)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

    # Add custom top layers for CIFAR-100 classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(512, activation='relu')(x)  # Fully connected layer
    x = Dropout(0.5)(x)  # Dropout for regularization
    x = Dense(num_classes, activation='softmax')(x)  # Output layer

    # Create the full model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),  # Adjust the learning rate if needed
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Add DenseNet-40 implementation
def dense_block(x, blocks, growth_rate):
    """Create a dense block with the given number of layers."""
    for i in range(blocks):
        # Bottleneck layer
        y = tf.keras.layers.BatchNormalization()(x)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(4 * growth_rate, 1, padding='same', use_bias=False)(y)
        
        # Conv layer
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)(y)
        
        x = tf.keras.layers.Concatenate()([x, y])
    return x

def transition_layer(x):
    """Create a transition layer between dense blocks."""
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(x.shape[-1], 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2)(x)
    return x

def create_densenet40(input_shape=(32, 32, 3), num_classes=10):
    """
    Create DenseNet-40 model as used in Guo et al. 2017.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
    Returns:
        tf.keras.Model: Configured DenseNet-40 model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False)(inputs)
    
    # Dense blocks with transitions
    for blocks in [12, 12, 12]:  # 3 dense blocks with 12 layers each
        x = dense_block(x, blocks, growth_rate=12)
        if blocks != 12:  # No transition after last block
            x = transition_layer(x)
    
    # Final layers
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile with the same settings as Guo et al.
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Add modern architectures
def create_resnet50(num_classes=10):
    """ResNet-50 with pre-trained weights"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_efficientnet_b0(num_classes=10):
    """EfficientNet-B0 with pre-trained weights"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_mnist_net(input_shape=(28, 28, 1), num_classes=10):
    """
    CNN Architecture for MNIST dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_fashion_net(input_shape=(28, 28, 1), num_classes=10):
    """
    CNN Architecture for Fashion MNIST dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(600, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cifar10_net(input_shape=(32, 32, 3), num_classes=10):
    """
    CNN Architecture for CIFAR-10 dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cifar100_net(input_shape=(32, 32, 3), num_classes=100):
    """
    CNN Architecture for CIFAR-100 dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    
def create_tiny_imagenet_net(input_shape=(64, 64, 3), num_classes=200):
    """
    CNN Architecture for Tiny ImageNet dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_pretrained_resnet50_tinyimagenet(input_shape=(64, 64, 3), num_classes=200):
    """
    Load a pretrained ResNet50 model and fine-tune it for Tiny ImageNet.
    """
    # Load the base ResNet50 model with pretrained weights on ImageNet
    base_model = ResNet50(
        include_top=False,  # Remove the original classification head
        weights="imagenet",  # Use ImageNet pretrained weights
        input_shape=input_shape
    )

    # Freeze the base model layers
    base_model.trainable = True

    # Add a new classification head for Tiny ImageNet
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Pooling layer
    x = tf.keras.layers.Dense(512, activation="relu")(x)  # Fully connected layer
    x = tf.keras.layers.Dropout(0.5)(x)  # Dropout for regularization
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)  # Final classifier
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def create_signlanguage_net(input_shape=(28, 28, 1), num_classes=24):
    """
    CNN Architecture for Sign Language dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_gtsrb_net(input_shape=(32, 32, 3), num_classes=43):
    """
    CNN Architecture for GTSRB dataset in Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


#RF configs
def create_RF_MNIST() :
    model = RandomForestClassifier(max_depth=45, max_features='sqrt', min_samples_split=5,
                        n_estimators=155 , random_state=0)
    return model


def create_RF_Fashion() :
    model = RandomForestClassifier(max_depth=23, max_features='sqrt', min_samples_split=10,
                        n_estimators=200, random_state=0)
    return model


def create_RF_GTSRB():
    """
    Random Forest configuration for GTSRB dataset.
    """
    model = RandomForestClassifier(
        max_depth=50,
        n_estimators=200,
        min_samples_split=5,
        random_state=0
    )
    return model

def create_RF_SignLanguage():
    """
    Random Forest configuration for Sign Language dataset.
    """
    model = RandomForestClassifier(
        max_depth=45,
        n_estimators=200,
        min_samples_split=5,
        random_state=0
    )
    return model

def create_RF_CIFAR() :
    model = RandomForestClassifier(max_depth=45, min_samples_split=5, n_estimators=200,
                        random_state=0)
    return model


def create_RF_KMNIST() :
    model = GradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_features='auto',
                            n_estimators=12, random_state=0)
    return model

def create_RF_CIFAR100():
    """
    Random Forest configuration for CIFAR-100 dataset.
    
    Returns:
        RandomForestClassifier: Configured for CIFAR-100.
    """
    model = RandomForestClassifier(
        n_estimators=300,          # More trees for robustness
        max_depth=50,              # Sufficient depth for complex patterns
        max_features='sqrt',       # Use square root of features to reduce correlation among trees
        min_samples_split=5,       # Controls overfitting by requiring at least 5 samples to split
        random_state=0             # Ensures reproducibility
    )
    return model


#GB configs

def create_GB_MNIST() :
    model = GradientBoostingClassifier(learning_rate=0.15, max_depth=12,
                            max_features='sqrt', n_estimators=70 , random_state=0)
    return model


def create_GB_Fashion() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=7, max_features='sqrt',
                            n_estimators=50, random_state=0)
    return model


def create_GB_GTSRB():
    """
    Gradient Boosting configuration for GTSRB dataset.
    """
    model = GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=8,
        max_features='sqrt',       # Reduces correlation between trees
        random_state=0             # Ensures reproducibility
    )
    return model

def create_GB_SignLanguage():
    """
    Gradient Boosting configuration for Sign Language dataset.
    """
    model = GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=8,
        max_features='sqrt',       # Reduces correlation between trees
        random_state=0
    )
    return model


def create_GB_CIFAR() :
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=8, max_features='sqrt',
                            n_estimators=20, random_state=0)
    return model


def create_GB_KMNIST() :
    model = GradientBoostingClassifier(learning_rate=0.15, max_depth=10,
                            max_features='sqrt', n_estimators=40,
                            random_state=0)
    return model

def create_GB_CIFAR100():
    """
    Gradient Boosting configuration for CIFAR-100 dataset.
    
    Returns:
        GradientBoostingClassifier: Configured for CIFAR-100.
    """
    model = GradientBoostingClassifier(
        n_estimators=200,          # More estimators to capture complex patterns
        learning_rate=0.1,         # Lower learning rate for better generalization
        max_depth=8,               # Depth to capture complexity without overfitting
        max_features='sqrt',       # Reduces correlation between trees
        validation_fraction=0.1,  # Use 10% of the data for validation
        n_iter_no_change=10,       # Stop if no improvement for 10 rounds
        random_state=0             # Ensures reproducibility
    )
    return model

