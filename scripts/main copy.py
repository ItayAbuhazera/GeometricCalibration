#!/usr/bin/env python
# coding: utf-8

import sys
import os

# Add the absolute path to the root directory of the project
sys.path.append("/cs/cs_groups/cliron_group/Calibrato")

import numpy as np
import tensorflow as tf
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from calibrators.geometric_calibrators import GeometricCalibrator
from utils.logging_config import setup_logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
from torchvision.models import DenseNet
import joblib
from models.model_factory import get_model  # Import the factory function
from tqdm import tqdm
import pandas as pd
import time
from utils.metrics import CalibrationMetrics


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Define the CNN model with flexible number of classes
def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Flexible num_classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define other models
def build_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_gb_model():
    return GradientBoostingClassifier(n_estimators=100, random_state=42)

def build_densenet40_model(num_classes=10):
    model = DenseNet(
        growth_rate=12,
        block_config=(6, 6, 6),
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        num_classes=num_classes
    )
    return model


def load_and_split_data(dataset_name, random_state):
    """
    Load and split data into train, validation, and test sets.
    
    Args:
        dataset_name (str): Name of the dataset to load ('MNIST', 'Fashion MNIST', 'CIFAR-10', 'CIFAR-100', 'GTRSB').
        random_state (int): Random state for reproducibility.
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    # Load dataset based on the given name
    if dataset_name.lower() == "mnist":
        (train_X_original, train_y_original), (test_X_original, test_y_original) = tf.keras.datasets.mnist.load_data()
        input_shape = (28, 28, 1)
    
    elif dataset_name.lower() == "fashion_mnist":
        (train_X_original, train_y_original), (test_X_original, test_y_original) = tf.keras.datasets.fashion_mnist.load_data()
        input_shape = (28, 28, 1)
    
    elif dataset_name.lower() == "cifar10":
        (train_X_original, train_y_original), (test_X_original, test_y_original) = tf.keras.datasets.cifar10.load_data()
        input_shape = (32, 32, 3)
    
    elif dataset_name.lower() == "cifar100":
        (train_X_original, train_y_original), (test_X_original, test_y_original) = tf.keras.datasets.cifar100.load_data()
        input_shape = (32, 32, 3)
    
    elif dataset_name.lower() == "gtrsb":
        # Placeholder for loading GTRSB (German Traffic Sign Benchmark)
        # Replace this with actual GTRSB loading code, e.g., from a local or custom dataset loader.
        # Example:
        # (train_X_original, train_y_original), (test_X_original, test_y_original) = load_gtrsb()
        raise NotImplementedError("GTRSB dataset loading not implemented. Use an appropriate data loader.")
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Choose from 'MNIST', 'Fashion MNIST', 'CIFAR-10', 'CIFAR-100', or 'GTRSB'.")

    # Combine train and test data for further splitting
    data = np.concatenate((train_X_original, test_X_original), axis=0)
    labels = np.concatenate((train_y_original, test_y_original), axis=0).squeeze()  # Ensure labels are 1D for compatibility

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    # Expand dimensions if necessary for grayscale images
    if input_shape[-1] == 1:  # Grayscale datasets (MNIST, Fashion MNIST)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

    # Ensure data is cast to float32 for neural network compatibility
    X_train, X_val, X_test = X_train.astype("float32"), X_val.astype("float32"), X_test.astype("float32")
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0

    return X_train, X_val, X_test, y_train, y_val, y_test



# Adjusted function to train or load the specified model type
def train_or_load_model(X_train, y_train, X_val, y_val, dataset_name, random_state, model_type="cnn", file_format="keras", epochs=None, epochs_dict=None):
    """
    Train or load a specified model type, with dynamic number of epochs based on dataset size.
    
    Args:
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data.
        y_val: Validation labels.
        dataset_name (str): Name of the dataset, used for directory structure.
        random_state (int): Random state number, used for directory structure.
        model_type (str): Type of model (e.g., "cnn").
        file_format (str): File format for saving model, options are "keras" or "h5".
        epochs (int, optional): If specified, use this number of epochs. Otherwise, it is dynamically calculated.
        epochs_dict (dict, optional): Dictionary mapping dataset names to specific epochs.
    
    Returns:
        model: Trained or loaded model.
    """
    model_directory = f"outputs/{dataset_name}/{random_state}/saved_models"
    model_path = os.path.join(model_directory, f"{model_type}_model.{file_format}")
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    # Determine the number of epochs
    if epochs_dict and dataset_name.lower() in epochs_dict:
        epochs = epochs_dict[dataset_name.lower()]
    elif epochs is None:
        # Dynamically calculate epochs based on the dataset size
        epochs = max(1, len(X_train) // 800)  # At least 1 epoch for very small datasets
    epochs = int(epochs)  # Ensure epochs is an integer

    logger.info(f"Using {epochs} epochs for training.")

    # Create the model
    model = get_model(dataset_name, model_type, input_shape=input_shape, num_classes=num_classes)

    # Load pre-trained model if exists
    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained {model_type} model from {model_path}.")
        if model_type == "cnn":
            model = tf.keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)
    else:
        # Train the model
        logger.info(f"Training new {model_type} model.")
        if model_type == "cnn":
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, verbose=2)
            os.makedirs(model_directory, exist_ok=True)
            model.save(model_path)
        else:
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            os.makedirs(model_directory, exist_ok=True)
            joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}.")

    return model

def preprocess_data(X):
    """
    Preprocess data for PyTorch. Adjust dimensions to match input requirements for the model.
    """
    X = torch.tensor(X, dtype=torch.float32)
    if len(X.shape) == 4 and X.shape[-1] == 1:  # Grayscale images
        X = X.permute(0, 3, 1, 2)  # Move channels to the first dimension
    elif len(X.shape) == 4 and X.shape[-1] == 3:  # RGB images
        X = X.permute(0, 3, 1, 2)  # Already in correct shape
    elif len(X.shape) == 3:  # Single channel but no explicit channel dimension
        X = X.unsqueeze(1)  # Add a channel dimension
    return X

def train_model_pytorch(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, learning_rate=0.001):
    """
    Train a PyTorch model with given training and validation data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training started on device: {device}")

    # Preprocess the training and validation data
    if not isinstance(X_train, torch.Tensor):
        X_train = preprocess_data(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(X_val, torch.Tensor):
        X_val = preprocess_data(X_val)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size
    )
    logger.info("DataLoaders initialized.")

    # Initialize metrics storage
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    logger.info(f"Starting training for {epochs} epochs.")
    for epoch in range(epochs):
        # Initialize tqdm for tracking progress
        epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        running_loss = 0.0
        model.train()
        for inputs, labels in epoch_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update tqdm with current batch loss
            epoch_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                _, preds = torch.max(outputs, 1)
                val_accuracy += torch.sum(preds == labels).item()

        val_loss /= len(val_loader)
        val_accuracy /= len(y_val)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Log progress for the epoch
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    logger.info("Training Complete!")
    logger.info(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")

    return history



def predict(model, data_loader, device):
    """
    Simulates a `predict` method for a PyTorch model.
    
    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for input data.
        device: Device on which to perform inference (e.g., 'cpu' or 'cuda').
        
    Returns:
        Numpy array of predictions.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Forward pass
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Select the class with highest probability
            predictions.append(preds.cpu().numpy())  # Collect predictions as a numpy array
            
    return np.concatenate(predictions, axis=0)

def calibrate_with_geometric(model, X_train, y_train, X_val, y_val, X_test, y_test, library):
    """
    Apply geometric calibration with the specified library (FAISS, KNN, or separation).
    """
    geo_calibrator = GeometricCalibrator(model=model, X_train=X_train, y_train=y_train, library=library)
    geo_calibrator.fit(X_val, y_val)

    # Calibrate the test set
    calibrated_probs = geo_calibrator.calibrate(X_test)
    y_test_pred = np.argmax(calibrated_probs, axis=1)
    accuracy = accuracy_score(y_test, y_test_pred)

    logger.info(f"Accuracy after calibration with {library}: {accuracy}")

    return calibrated_probs, y_test_pred

def calculate_ece(probs, y_pred, y_true, n_bins=20):
    """
    Calculate Expected Calibration Error (ECE).
    """
    confidence_of_pred_class = np.max(probs, axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1

    total_error = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_confidences = confidence_of_pred_class[bin_mask]
        bin_real = y_true[bin_mask]
        bin_pred = y_pred[bin_mask]

        if len(bin_confidences) > 0:
            bin_acc = np.mean(bin_real == bin_pred)
            bin_conf = np.mean(bin_confidences)
            bin_weight = len(bin_confidences) / len(probs)
            total_error += bin_weight * np.abs(bin_acc - bin_conf)

    logger.info(f"Final ECE value: {total_error}")
    return total_error


def main(dataset_name, random_state, model_type="cnn", technique="faiss"):
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(dataset_name, random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dictionary mapping datasets to epoch counts
    dataset_epochs = {
        "mnist": 10,
        "fashion_mnist": 20,
        "cifar10": 35,
        "cifar100": 50,
        "sign_language": 25,
    }
    # Train or load the model based on user selection
    model = train_or_load_model(
        X_train, y_train, X_val, y_val, 
        dataset_name=dataset_name, 
        random_state=random_state, 
        model_type="cnn", 
        epochs_dict=dataset_epochs
    )

    # Initialize results for pandas DataFrame
    results = []

    # Get dataset sizes
    train_size = len(X_train)
    val_size = len(X_val)
    test_size = len(X_test)

    # Get features for validation and test sets
    if model_type in ["cnn", "densenet"]:
        features_test = model.predict(X_test)
        y_test_pred = np.argmax(features_test, axis=1)
    else:
        features_test = model.predict_proba(X_test.reshape(X_test.shape[0], -1))
        y_test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))


    try:
        start_time = time.time()
        calibrated_probs, y_test_pred_cal = calibrate_with_geometric(
            model, X_train, y_train, X_val, y_val, X_test, y_test, library=technique
        )
        calibration_time = time.time() - start_time
    except Exception as e:
        print(f"Error during calibration with technique '{technique}': {e}")
        return
    
    # Calculate metrics
    metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
    metrics_dict = metrics.calculate_all_metrics()
    
    # Uncalibrated metrics
    metrics_uncalibrated = CalibrationMetrics(features_test, y_test_pred, y_test, n_bins=20)
    metrics_dict_uncalibrated = metrics_uncalibrated.calculate_all_metrics()
    results.append({
        "Metric": "Uncalibrated",
        **metrics_dict_uncalibrated,
        "Calibration Time (s)": "N/A",
        "Train Size": train_size,
        "Validation Size": val_size,
        "Test Size": test_size
    })
    print(f"Uncalibrated Metrics: {metrics_dict_uncalibrated}")

    # Create the directory structure
    base_dir = f"/cs/cs_groups/cliron_group/Calibrato/{dataset_name}/{random_state}/{technique}/{model_type}"
    os.makedirs(base_dir, exist_ok=True)  # Create directories if they don't exist

    # Save metrics to a CSV file in the desired location
    csv_file = os.path.join(base_dir, "results.csv")
    results = [{
        "Metric": technique,
        **metrics_dict,
        "Calibration Time (s)": calibration_time,
        "Train Size": train_size,
        "Validation Size": val_size,
        "Test Size": test_size
    }]

    # Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Save DataFrame to a CSV file
    csv_file = f"results_{dataset_name}_randomstate_{random_state}.csv"
    results_df.to_csv(csv_file, index=False)

    print(f"Results saved to {csv_file}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration experiment with a specific technique.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., mnist, cifar10).")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for reproducibility.")
    parser.add_argument("--model_type", type=str, default="cnn", help="Model type (e.g., cnn, densenet).")
    parser.add_argument("--technique", type=str, required=True, help="Calibration technique (faiss, knn, separation).")
    args = parser.parse_args()

    main(args.dataset_name, args.random_state, args.model_type, args.technique)

