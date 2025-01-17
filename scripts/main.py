#!/usr/bin/env python
# coding: utf-8

import sys
import os
from PIL import Image

# Add the absolute path to the root directory of the project
sys.path.append("/cs/cs_groups/cliron_group/Calibrato")
import time
import io

import numpy as np
import tensorflow as tf
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from calibrators.geometric_calibrators import GeometricCalibrator, GeometricCalibratorTrust
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
from utils.utils import StabilitySpace, Compression
import random
from skimage.transform import rotate
from scipy.ndimage import shift
# from calibrators.calibrators import *
from calibrators.ensemble_calibrators import *
from calibrators.non_parametric_calibrators import *
from calibrators.parametric_calibrators import *
from calibrators.specialized_calibrators import *
from calibrators.trust_score_calibration import *
from sklearn.calibration import CalibratedClassifierCV
from filelock import FileLock
from sklearn.neighbors import KDTree, NearestNeighbors  # Add this line



# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def transform_test_set(X_test, transform_ratios=(0.3, 0.3, 0.4), random_state=None):
    """
    Apply transformations to the test set: rotate, shift, and add noise.

    Args:
        X_test (numpy.ndarray): Test set images.
        transform_ratios (tuple): Ratios for (rotation, shift, noise).
        random_state (int): Seed for reproducibility.

    Returns:
        numpy.ndarray: Transformed test set.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X_test)
    n_rotate = int(n_samples * transform_ratios[0])
    n_shift = int(n_samples * transform_ratios[1])
    n_noise = n_samples - n_rotate - n_shift

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Get subsets
    rotate_indices = indices[:n_rotate]
    shift_indices = indices[n_rotate:n_rotate + n_shift]
    noise_indices = indices[n_rotate + n_shift:]

    # Apply transformations
    transformed_X_test = X_test.copy()

    # Rotation
    for idx in rotate_indices:
        angle = random.uniform(-30, 30)  # Rotate by a random angle between -30 and 30 degrees
        transformed_X_test[idx] = rotate(transformed_X_test[idx], angle, mode='wrap')

    # Shifting
    for idx in shift_indices:
        shift_x = random.uniform(-5, 5)  # Shift up to ±5 pixels
        shift_y = random.uniform(-5, 5)
        transformed_X_test[idx] = shift(transformed_X_test[idx], shift=(shift_x, shift_y, 0), mode='wrap')

    # Noise
    for idx in noise_indices:
        noise = np.random.random(transformed_X_test[idx].shape) * 0.4  # Add up to 40% noise
        transformed_X_test[idx] = np.clip(transformed_X_test[idx] + noise, 0, 1)

    return transformed_X_test


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

def load_and_split_gtsrb_data(train_csv_path, test_csv_path, random_state):
    """
    Load GTSRB data from CSV files, concatenate, and split into train, validation, and test sets.
    
    Args:
        train_csv_path (str): Path to the train.csv file.
        test_csv_path (str): Path to the test.csv file.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Load CSV files
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Initialize lists to hold image data and labels
    images = []
    labels = []
    
    # Function to load and preprocess images
    def load_images_from_df(df, base_path):
        for _, row in df.iterrows():
            img_path = os.path.join(base_path, row['Path'])
            try:
                image = Image.open(img_path).convert("RGB").resize((32, 32))
                images.append(np.array(image))
                labels.append(row['ClassId'])
            except Exception as e:
                logging.error(f"Error loading image {img_path}: {e}")
    
    # Load images and labels from train and test data
    logging.info("Loading training images...")
    load_images_from_df(train_df, os.path.dirname(train_csv_path))
    logging.info("Loading test images...")
    load_images_from_df(test_df, os.path.dirname(test_csv_path))
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels, dtype=np.int64)
    logging.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split_signlanguage_data(train_csv_path, test_csv_path, random_state): 
    """
    Load Sign Language data from CSV files, concatenate, and split into train, validation, and test sets.
    """
    logging.info(f"Loading Sign Language dataset from {train_csv_path} and {test_csv_path}.")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Replace label 24 with 9
    logging.info("Replacing label 24 with 9.")
    train_df["label"].replace({24: 9}, inplace=True)
    test_df["label"].replace({24: 9}, inplace=True)

    # Separate labels from features
    train_y = train_df.pop('label').values
    test_y = test_df.pop('label').values

    # Convert features to numpy arrays
    train_X = train_df.to_numpy(dtype=np.float32)
    test_X = test_df.to_numpy(dtype=np.float32)

    # Normalize features to the range [0, 1]
    logging.info("Normalizing feature values.")
    train_X /= 255.0
    test_X /= 255.0

    # Reshape data to include the channel dimension (1 for grayscale images)
    train_X = train_X.reshape((-1, 28, 28, 1))
    test_X = test_X.reshape((-1, 28, 28, 1))

    # Combine train and test data for further splitting
    logging.info("Combining train and test data for splitting.")
    data = np.concatenate((train_X, test_X), axis=0)
    labels = np.concatenate((train_y, test_y), axis=0)

    # Split data into train, validation, and test sets
    logging.info("Splitting data into train, validation, and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split_data(dataset_name, random_state):
    """
    Load and split data into train, validation, and test sets.
    
    Args:
        dataset_name (str): Name of the dataset to load ('MNIST', 'Fashion MNIST', 'CIFAR-10', 'CIFAR-100').
        random_state (int): Random state for reproducibility.
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logging.info(f"Loading dataset: {dataset_name}")
    
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

    # elif dataset_name.lower() == "tiny_imagenet":
    #     logging.info("Loading Tiny ImageNet dataset")
    #     splits = {
    #         'train': 'data/train-00000-of-00001-1359597a978bc4fa.parquet',
    #         'valid': 'data/valid-00000-of-00001-70d52db3c749a935.parquet'
    #     }
        
    #     # Load training and validation data
    #     train_df = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits['train'])
    #     valid_df = pd.read_parquet("hf://datasets/zh-plus/tiny-imagenet/" + splits['valid'])
        
    #     logging.info(f"Train DataFrame head:\n{train_df.head()}")
    #     logging.info(f"Validation DataFrame head:\n{valid_df.head()}")
        
    #     # Decode images from binary data
    #     def decode_image(row):
    #         binary_data = row['bytes']  # Extract binary data
    #         try:
    #             image = Image.open(io.BytesIO(binary_data))  # Decode the image
    #             image = image.convert("RGB")  # Ensure RGB
    #             image = image.resize((64, 64))  # Resize to 64x64
    #             return np.array(image)  # Convert to numpy array
    #         except Exception as e:
    #             logging.error(f"Failed to decode or resize image: {e}")
    #             return None

    #     # Decode and filter valid images
    #     train_X_original = np.stack([
    #         img for img in (decode_image(img) for img in train_df['image']) if img is not None
    #     ])
    #     test_X_original = np.stack([
    #         img for img in (decode_image(img) for img in valid_df['image']) if img is not None
    #     ])
    #     logging.info(f"Shape of train_X_original: {train_X_original.shape}")
    #     logging.info(f"Shape of test_X_original: {test_X_original.shape}")

    #     train_y_original = np.array(train_df['label'])
    #     test_y_original = np.array(valid_df['label'])

    #     # Set random seed for reproducibility (optional)
    #     np.random.seed(random_state)

    #     # Calculate the number of samples for half the dataset
    #     num_samples_train = train_X_original.shape[0] // 4
    #     num_samples_test = test_X_original.shape[0] // 2

    #     # Generate random indices to sample
    #     random_indices_train = np.random.choice(train_X_original.shape[0], size=num_samples_train, replace=False)
    #     random_indices_test = np.random.choice(test_X_original.shape[0], size=num_samples_test, replace=False)

    #     # Sample the images and labels using the random indices
    #     train_X_original = train_X_original[random_indices_train]
    #     train_y_original = train_y_original[random_indices_train]
    #     test_X_original = test_X_original[random_indices_test]
    #     test_y_original = test_y_original[random_indices_test]

    #     logging.info(f"Shape of sampled train_X: {train_X_original.shape}, train_y: {train_y_original.shape}")
    #     logging.info(f"Shape of sampled test_X: {test_X_original.shape}, test_y: {test_y_original.shape}")

        
    #     input_shape = (64, 64, 3)  # Tiny ImageNet images are 64x64 RGB
    #     logging.info("Tiny ImageNet loaded successfully")


    elif dataset_name.lower() == "gtsrb":
        train_csv_path = '/cs/cs_groups/cliron_group/Calibrato/GTSRB/data/Train.csv'
        test_csv_path = '/cs/cs_groups/cliron_group/Calibrato/GTSRB/data/Test.csv'
        return load_and_split_gtsrb_data(train_csv_path, test_csv_path, random_state)
    
    elif dataset_name.lower() == "signlanguage":
        train_csv_path = '/cs/cs_groups/cliron_group/Calibrato/SignLanguage/data/sign_mnist_train.csv'
        test_csv_path = '/cs/cs_groups/cliron_group/Calibrato/SignLanguage/data/sign_mnist_test.csv'
        return load_and_split_signlanguage_data(train_csv_path, test_csv_path, random_state)
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Choose from 'MNIST', 'Fashion MNIST', 'CIFAR-10', 'CIFAR-100', or 'GTSRB'.")

    logging.info("Combining and splitting data")
    
    # Combine train and test data for further splitting
    data = np.concatenate((train_X_original, test_X_original), axis=0)
    labels = np.concatenate((train_y_original, test_y_original), axis=0).squeeze()  # Ensure labels are 1D for compatibility
    logging.debug(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    logging.debug(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Expand dimensions if necessary for grayscale images
    if input_shape[-1] == 1:  # Grayscale datasets (MNIST, Fashion MNIST)
        logging.info("Expanding dimensions for grayscale images")
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

    # Ensure data is cast to float32 for neural network compatibility
    X_train, X_val, X_test = X_train.astype("float32"), X_val.astype("float32"), X_test.astype("float32")
    X_train /= 255.0
    X_val /= 255.0
    X_test /= 255.0
    logging.info("Data normalization completed")

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
    model_directory = f"{dataset_name}/{random_state}/saved_models"
    model_path = os.path.join(model_directory, f"{model_type}_model.{file_format}")
    lock_path = f"{model_path}.lock"
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

    # Preprocess data for ResNet if needed
    if model_type == "pretrained_resnet":
        batch = 16
    else:
        batch = 32
        
    # Use a file lock to handle concurrent access
    with FileLock(lock_path):
        # Check if the model already exists
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained {model_type} model from {model_path}.")
            if model_type in ["cnn", "pretrained_resnet", "pretrained_efficientnet"]:
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
        else:
            # Train the model
            logger.info(f"Training new {model_type} model.")
            model = get_model(dataset_name, model_type, input_shape=input_shape, num_classes=num_classes)

            if model_type in ["cnn", "pretrained_resnet", "pretrained_efficientnet"]:
                model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch, verbose=2)
                os.makedirs(model_directory, exist_ok=True)
                model.save(model_path)
                logger.info(f"Model saved at {model_path}.")
                # Exit after training the CNN model
                logger.info("Exiting program after training the CNN model to free GPU resources.")
                sys.exit(0)
            else:
                model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
                os.makedirs(model_directory, exist_ok=True)
                joblib.dump(model, model_path)

            logger.info(f"Model saved at {model_path}.")
    
    return model



def calibrate_with_geometric(model, X_train, y_train, X_val, y_val, X_test, y_test, library, metric='l2'):
    """
    Apply geometric calibration with the specified library (FAISS, KNN, or separation).
    
    Args:
        model: The model to calibrate
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        library: Library to use for stability calculation
        metric: Distance metric to use (default: 'l2')
    """
    geo_calibrator = GeometricCalibrator(
        model=model, 
        X_train=X_train, 
        y_train=y_train, 
        library=library,
        metric=metric
    )
    geo_calibrator.fit(X_val, y_val)

    # Calibrate the test set
    calibrated_probs = geo_calibrator.calibrate(X_test)
    y_test_pred = np.argmax(calibrated_probs, axis=1)
    accuracy = accuracy_score(y_test, y_test_pred)

    logger.info(f"Accuracy after calibration with {library} using {metric} metric: {accuracy}")

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

def calibrate_specialized(features_val, y_val, features_test, y_test, calibrator, name, technique_dirs, y_test_pred):
    """
    Perform specialized calibration (SBC, HB, BBQ, Beta) and save results.

    Args:
        features_val (ndarray): Validation set predictions.
        y_val (ndarray): Validation set labels.
        features_test (ndarray): Test set predictions.
        y_test (ndarray): Test set labels.
        calibrator (object): Specialized calibrator instance.
        name (str): Name of the calibration method.
        technique_dirs (dict): Directories to save calibration results.

    Returns:
        dict: A dictionary containing calibration metrics.
    """
    try:
        logger.info(f"Starting specialized calibration: {name}.")

        # Step 1: Fit the calibrator using the validation set
        logger.info(f"Fitting {name} calibrator.")
        calibrator.fit(features_val, y_val)

        # Step 2: Calibrate the test set
        logger.info(f"Calibrating test set with {name}.")
        start_time = time.time()
        calibrated_probs = calibrator.calibrate(features_test)
        calibration_time = time.time() - start_time

        # Step 3: Compute predicted classes
        if calibrated_probs.ndim == 1:  # If 1D, reshape to 2D for binary classification
            y_test_pred_cal = y_test_pred
        else:
            y_test_pred_cal = np.argmax(calibrated_probs, axis=0)

        # Step 4: Calculate metrics using CalibrationMetrics
        logger.info(f"Calculating metrics for {name}.")
        metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
        metrics_dict = metrics.calculate_all_metrics()

        # Step 5: Prepare and save results
        results = {
            "Metric": name.replace("_", " ").capitalize(),
            **metrics_dict,
            "Calibration Time (s)": calibration_time
        }

        results_csv_file = os.path.join(technique_dirs[name], "results.csv")
        pd.DataFrame([results]).to_csv(results_csv_file, index=False)

        logger.info(f"{name.capitalize()} Metrics saved to {results_csv_file}.")
        return results

    except Exception as e:
        logger.error(f"Error during specialized calibration ({name}): {e}", exc_info=True)
        return None

def initialize_directories(base_dir, transformed, dataset_name, random_state, model_type, metric, trust_alpha):
    if transformed:
        base_dir = base_dir + "/transformed"
    os.makedirs(base_dir, exist_ok=True)

    technique_dirs = {
        "faiss_exact": os.path.join(base_dir, "faiss_exact"),
        "faiss_exact_binned": os.path.join(base_dir, "faiss_exact_binned"),
        "knn_binned": os.path.join(base_dir, "knn_binned"),
        "knn": os.path.join(base_dir, "knn"),
        "separation": os.path.join(base_dir, "separation"),
        "isotonic": os.path.join(base_dir, "isotonic"),
        "platt": os.path.join(base_dir, "platt"),
        "temperature": os.path.join(base_dir, "temperature"),
        "trust_score_filtered": os.path.join(base_dir, f"trust_score_filtered"),
        "geometric_trust_binned": os.path.join(base_dir, f"geometric_trust_binned"),
        "trust_score_unfiltered": os.path.join(base_dir, f"trust_score_unfiltered"),
        "kdtree": os.path.join(base_dir, "kdtree"),
        "seperation": os.path.join(base_dir, "seperation"),
        # Add new directories:
        "sbc": os.path.join(base_dir, "sbc"),
        "hb": os.path.join(base_dir, "hb"),
        "bbq": os.path.join(base_dir, "bbq"), 
        "beta": os.path.join(base_dir, "beta"),
        "ets": os.path.join(base_dir,"ets")
    }

    all_results_dir = os.path.join(base_dir, "all")
    os.makedirs(all_results_dir, exist_ok=True)

    for directory in technique_dirs.values():
        os.makedirs(directory, exist_ok=True)

    return base_dir, technique_dirs, all_results_dir


def prepare_data(dataset_name, random_state, transformed):
    """Load and optionally transform dataset."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(dataset_name, random_state)
    if transformed:
        X_test = transform_test_set(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_model(X_train, y_train, X_val, y_val, dataset_name, random_state, model_type, dataset_epochs):
    """Train or load the model."""
    return train_or_load_model(
        X_train, y_train, X_val, y_val,
        dataset_name=dataset_name,
        random_state=random_state,
        model_type=model_type,
        epochs_dict=dataset_epochs
    )

def calibrate_geometric(model, X_train, y_train, X_val, y_val, X_test, y_test, calibrator, name, technique_dirs, metric, use_binning, compression=None):
    """
    Perform geometric calibrations (FAISS, KNN, Separation) with optional binning.
    """
    try:
        # Apply compression if provided
        stability_space = StabilitySpace(
            X_train, 
            y_train, 
            compression=compression,
            library=calibrator["library"],
            faiss_mode=calibrator.get("mode"), 
            metric=metric
        )

        geo_calibrator = GeometricCalibrator(
            model=model,
            X_train=X_train,
            y_train=y_train,
            stability_space=stability_space,
            library=calibrator["library"],
            metric=metric,
            use_binning=use_binning,  # Enable or disable binning
        )
        geo_calibrator.fit(X_val, y_val)
        start_time = time.time()
        calibrated_probs = geo_calibrator.calibrate(X_test)
        calibration_time = time.time() - start_time
        samples_per_second = len(X_test) / calibration_time
        y_test_pred_cal = np.argmax(calibrated_probs, axis=1)

        metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
        metrics_dict = metrics.calculate_all_metrics()

        results = {
            "Metric": f"{name.replace('_', ' ').capitalize()}",
            **metrics_dict,
            "Calibration Time (s)": calibration_time,
            "Samples per Second": samples_per_second,
        }

        results_csv_file = os.path.join(technique_dirs[name], "results.csv")
        pd.DataFrame([results]).to_csv(results_csv_file, index=False)
        print(f"{name.replace('_', ' ').capitalize()} Metrics saved to {results_csv_file}")
        return results
    except Exception as e:
        logger.error(f"Error calibrating with {name}: {e}")
        return None

def calibrate_parametric(features_val, y_val, features_test, y_test, calibrator, name, technique_dirs):
    try:
        logger.info(f"Starting parametric calibration with {name}.")
        calibrator.fit(features_val, y_val)
        start_time = time.time()
        calibrated_probs = calibrator.calibrate(features_test)
        calibration_time = time.time() - start_time
        samples_per_second = len(features_test) / calibration_time

        # Ensure calibrated_probs is 2D
        if calibrated_probs.ndim == 1:
            # Convert to 2D array with two classes (binary classification)
            calibrated_probs = np.vstack([1 - calibrated_probs, calibrated_probs]).T

        y_test_pred_cal = np.argmax(calibrated_probs, axis=1)

        metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
        metrics_dict = metrics.calculate_all_metrics()

        results = {
            "Metric": name.replace("_", " ").capitalize(),
            **metrics_dict,
            "Calibration Time (s)": calibration_time,
            "Samples per Second": samples_per_second,
        }

        results_csv_file = os.path.join(technique_dirs[name], "results.csv")
        pd.DataFrame([results]).to_csv(results_csv_file, index=False)
        print(f"{name.replace('_', ' ').capitalize()} Metrics saved to {results_csv_file}")
        return results
    except Exception as e:
        logger.error(f"Error calibrating with {name}: {e}")
        return None

def calibrate_trust_score(model, X_train, y_train, X_val, y_val, X_test, y_test, trust_alpha, technique_dirs, use_filtering=True, use_binning=True, n_bins=50):
    """
    Calibrates a model using the Geometric Trust Score method and saves the results to the appropriate directory.

    Args:
        model: The model to calibrate.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
        trust_alpha: The alpha value for trust score calibration.
        technique_dirs: Dictionary of directories for saving results.
        use_filtering (bool): Whether to filter the trust scores.
        use_binning (bool): Whether to bin the trust scores.
        n_bins (int): Number of bins to use for binning.
    Returns:
        dict: A dictionary containing calibration metrics.
    """
    try:
        logger.info(f"Starting Geometric Trust Score calibration with binning: {use_binning}, filtering: {use_filtering}")
        
        # Log original shapes
        logger.info(f"Shapes before flattening: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
        # Handle input reshaping based on model type
        if isinstance(model, tf.keras.Model):
            X_train_flat, X_val_flat, X_test_flat = X_train, X_val, X_test
            logger.info(f"Model type: TensorFlow/Keras. Model input shape: {model.input_shape}")
        else:
            X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
            X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val
            X_test_flat = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
            logger.info(f"Model type: {type(model).__name__}. Data reshaped for scikit-learn model.")

        # Log flattened shapes
        logger.info(f"Shapes after flattening: X_train_flat: {X_train_flat.shape}, X_val_flat: {X_val_flat.shape}, X_test_flat: {X_test_flat.shape}")

        # Select appropriate directory for saving results
        technique_key = f"trust_score_{'filtered' if use_filtering else 'unfiltered'}"
        if use_binning:
            technique_key = f"geometric_trust_binned"
        if technique_key not in technique_dirs:
            raise ValueError(f"Technique directory for '{technique_key}' not found.")

        results_dir = technique_dirs[technique_key]
        results_csv_file = os.path.join(results_dir, "results.csv")

        # Check if results already exist
        if os.path.exists(results_csv_file):
            logger.info(f"Results already exist for {technique_key}. Skipping calibration.")
            return None

        # Initialize and fit Geometric Trust Score Calibrator
        logger.info(f"Initializing GeometricCalibratorTrust with use_filtering={use_filtering}, use_binning={use_binning}, n_bins={n_bins}")
        geometric_trust = GeometricCalibratorTrust(
            model=model,
            X_train=X_train_flat,
            y_train=y_train,
            k=10,
            min_dist=1e-12,
            use_binning=use_binning,
            n_bins=n_bins,
            use_filtering=use_filtering,
            alpha=trust_alpha
        )
        
        logger.info("Fitting Geometric Trust Score Calibrator using validation set.")
        logger.info(f"Predicting with X_val of shape: {X_val_flat.shape}")
        geometric_trust.fit(X_val_flat, y_val)
        
        logger.info("Fitting completed. Calibrating the test set.")
        start_time = time.time()
        calibrated_probs = geometric_trust.calibrate(X_test_flat)
        calibration_time = time.time() - start_time
        samples_per_second = len(X_test_flat) / calibration_time
        
        logger.info("Evaluating calibrated results.")
        y_test_pred_cal = np.argmax(calibrated_probs, axis=1)
        metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
        metrics_dict = metrics.calculate_all_metrics()
        
        results = {
            "Metric": f"Geometric Trust Score {'Binned' if use_binning else 'Unbinned'}",
            **metrics_dict,
            "Calibration Time (s)": calibration_time,
            "Samples per Second": samples_per_second,
        }
        
        # Save results to the appropriate directory
        pd.DataFrame([results]).to_csv(results_csv_file, index=False)
        logger.info(f"Geometric Trust Score {'Binned' if use_binning else 'Unbinned'} Metrics saved to {results_csv_file}")
        return results

    except Exception as e:
        logger.error(f"Error during Geometric Trust Score calibration: {e}", exc_info=True)
        return None    


def compute_uncalibrated_metrics(features_test, y_test_pred, y_test, train_size, val_size, test_size):
    """Compute metrics for the uncalibrated model."""
    metrics_uncalibrated = CalibrationMetrics(features_test, y_test_pred, y_test, n_bins=20)
    metrics_dict = metrics_uncalibrated.calculate_all_metrics()
    results = {
        "Metric": "Uncalibrated",
        **metrics_dict,
        "Calibration Time (s)": "N/A",
        "Train Size": train_size,
        "Validation Size": val_size,
        "Test Size": test_size
    }
    print(f"Uncalibrated Metrics: {metrics_dict}")
    return results


def save_results(results, existing_results, all_results_dir):
    """
    Combine new and existing results, then save to a CSV file.

    Args:
        results (list): List of new result dictionaries.
        existing_results (list): List of existing result DataFrames.
        all_results_dir (str): Directory where the combined results should be saved.
    """
    # Convert new results to a DataFrame
    new_results_df = pd.DataFrame(results)
    
    # Combine with existing results if any
    if existing_results:
        # Combine existing dataframes directly without re-wrapping
        combined_results = pd.concat(existing_results + [new_results_df], ignore_index=True)
    else:
        combined_results = new_results_df
    
    # Save combined results to a CSV file
    results_csv_file = os.path.join(all_results_dir, "all_results.csv")
    combined_results.to_csv(results_csv_file, index=False)
    
    logger.info(f"All results combined and saved to {results_csv_file}")
    print(f"All results combined and saved to {results_csv_file}")


def handle_existing_results(name, technique_dirs, existing_results):
    results_csv_file = os.path.join(technique_dirs[name], "results.csv")
    if os.path.exists(results_csv_file):
        logger.info(f"Found existing results for {name}. Adding to final results.")
        existing_results.append(pd.read_csv(results_csv_file))
        return True
    return False

def perform_trust_score_calibration(calibrator, model, X_train, y_train, X_val, y_val, X_test, y_test, trust_alpha, technique_dirs):
    use_filtering = calibrator["use_filtering"]
    return calibrate_trust_score(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trust_alpha=trust_alpha,
        technique_dirs=technique_dirs,
        use_filtering=use_filtering
    )

def perform_parametric_calibration(calibrator, features_val, y_val, features_test, y_test, name, technique_dirs):
    return calibrate_parametric(features_val, y_val, features_test, y_test, calibrator, name, technique_dirs)

def perform_specialized_calibration(calibrator, features_val, y_val, features_test, y_test, name, technique_dirs, y_test_pred):
    """
    Perform specialized calibration (SBC, HB, BBQ, Beta) and save results.

    Args:
        features_val (ndarray): Validation set predictions.
        y_val (ndarray): Validation set labels.
        features_test (ndarray): Test set predictions.
        y_test (ndarray): Test set labels.
        calibrator (object): Specialized calibrator instance.
        name (str): Name of the calibration method.
        technique_dirs (dict): Directories to save calibration results.

    Returns:
        dict: A dictionary containing calibration metrics.
    """
    try:
        logger.info(f"Starting specialized calibration: {name}.")

        # Step 1: Fit the calibrator using the validation set
        logger.info(f"Fitting {name} calibrator.")
        calibrator.fit(features_val, y_val)

        # Step 2: Calibrate the test set
        logger.info(f"Calibrating test set with {name}.")
        start_time = time.time()
        calibrated_probs = calibrator.calibrate(features_test)
        calibration_time = time.time() - start_time
        samples_per_second = len(features_test) / calibration_time

        # Step 3: Compute predicted classes
        if calibrated_probs.ndim == 1:  # If 1D, reshape to 2D for binary classification
            y_test_pred_cal = y_test_pred
        else:
            y_test_pred_cal = np.argmax(calibrated_probs, axis=0)

        # Step 4: Calculate metrics using CalibrationMetrics
        logger.info(f"Calculating metrics for {name}.")
        metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
        metrics_dict = metrics.calculate_all_metrics()

        # Step 5: Prepare and save results
        results = {
            "Metric": name.replace("_", " ").capitalize(),
            **metrics_dict,
            "Calibration Time (s)": calibration_time,
            "Samples per Second": samples_per_second
        }

        results_csv_file = os.path.join(technique_dirs[name], "results.csv")
        pd.DataFrame([results]).to_csv(results_csv_file, index=False)

        logger.info(f"{name.capitalize()} Metrics saved to {results_csv_file}.")
        return results

    except Exception as e:
        logger.error(f"Error during specialized calibration ({name}): {e}", exc_info=True)
        return None

def perform_ensemble_ts_calibration(calibrator, features_val, y_val, features_test, y_test):
    logger.info("Running EnsembleTSCalibrator.")
    calibrator.fit(features_val, y_val)
    start_time = time.time()
    calibrated_probs = calibrator.calibrate(features_test)
    calibration_time = time.time() - start_time
    samples_per_second = len(features_test) / calibration_time
    y_test_pred_cal = np.argmax(calibrated_probs, axis=1)
    metrics = CalibrationMetrics(calibrated_probs, y_test_pred_cal, y_test, n_bins=20)
    return {
        "Metric": "EnsembleTS",
        **metrics.calculate_all_metrics(),
        "Calibration Time (s)": calibration_time,
        "Samples per Second": samples_per_second
    }

def perform_geometric_calibration(calibrator, model, X_train, y_train, X_val, y_val, X_test, y_test, name, technique_dirs, metric, compression):
    use_binning = calibrator.get("binned", False)
    return calibrate_geometric(
        model, X_train, y_train, X_val, y_val, X_test, y_test, calibrator, name, technique_dirs, metric, use_binning, compression
    )

def perform_calibrations(calibrations, model, X_train, y_train, X_val, y_val, X_test, y_test, features_val, features_test, y_test_pred, technique_dirs, trust_alpha, metric, compression):
    results = []
    existing_results = []
    for name, calibrator in calibrations.items():
        if name not in technique_dirs:
            logger.error(f"Technique directory for '{name}' not found. Skipping this calibration method.")
            continue

        if handle_existing_results(name, technique_dirs, existing_results):
            continue

        try:
            if name.startswith("trust_score"):
                trust_results = perform_trust_score_calibration(calibrator, model, X_train, y_train, X_val, y_val, X_test, y_test, trust_alpha, technique_dirs)
                if trust_results:
                    results.append(trust_results)
            elif name in ["isotonic", "platt", "temperature"]:
                parametric_results = perform_parametric_calibration(calibrator, features_val, y_val, features_test, y_test, name, technique_dirs)
                if parametric_results:
                    results.append(parametric_results)
            elif name in ["sbc", "hb", "bbq", "beta"]:
                specialized_results = perform_specialized_calibration(calibrator, features_val, y_val, features_test, y_test, name, technique_dirs, y_test_pred)
                if specialized_results:
                    results.append(specialized_results)
            elif name == "ets":
                ensemble_ts_results = perform_ensemble_ts_calibration(calibrator, features_val, y_val, features_test, y_test)
                if ensemble_ts_results:
                    results.append(ensemble_ts_results)
            else:
                geometric_results = perform_geometric_calibration(calibrator, model, X_train, y_train, X_val, y_val, X_test, y_test, name, technique_dirs, metric, compression)
                if geometric_results:
                    results.append(geometric_results)
        except Exception as e:
            logger.error(f"Error during calibration method '{name}': {e}")
            continue
    return results, existing_results

def main(dataset_name, random_state, model_type="cnn", metric="L2", transformed=False, trust_alpha=0.1,
         compression_types=None, compression_params=2):
    
    base_path = f"/cs/cs_groups/cliron_group/Calibrato/{dataset_name}/{random_state}/{model_type}/{metric}"
    if transformed:
        base_path = os.path.join(base_path, "transformed")
    all_results_path = os.path.join(base_path, "all")
    for _, _, files in os.walk(all_results_path):
        for file in files:
            if file.endswith("all_results.csv"):
                logger.info(f"Results file found at {os.path.join(all_results_path, file)}.")
                return

    # Initialize directories
    base_dir, technique_dirs, all_results_dir = initialize_directories(
        f"/cs/cs_groups/cliron_group/Calibrato/{dataset_name}/{random_state}/{model_type}/{metric}",
        transformed, dataset_name, random_state, model_type, metric, trust_alpha
    )
    compression = None
    if compression_types:
        logger.info(f"Initializing Compression with types: {compression_types}, params: {compression_params}")
        compression = Compression(compression_types=compression_types, compression_params=compression_params)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(dataset_name, random_state, transformed)

    # Train or load model
    dataset_epochs = {
        "mnist": 10,
        "fashion_mnist": 20,
        "cifar10": 35,
        "cifar100": 75,
        "sign_language": 25,
        # "tiny_imagenet": 30,
        "gtsrb": 20,
    }
    model = prepare_model(X_train, y_train, X_val, y_val, dataset_name, random_state, model_type, dataset_epochs)

    # Get dataset sizes
    train_size, val_size, test_size = len(X_train), len(X_val), len(X_test)

    # Preprocess data based on the model type
    if model_type in ["cnn", "densenet", "pretrained_resnet", "pretrained_efficientnet"]:
        logger.info(f"Using TensorFlow/Keras model: {model_type}")
        features_test = model.predict(X_test)
        y_test_pred = np.argmax(features_test, axis=1)
        features_val = model.predict(X_val)
        y_val_pred = np.argmax(features_val, axis=1)
    else:
        logger.info(f"Using sklearn model: {model_type}")
        X_test = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
        X_val = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val

        features_test = model.predict_proba(X_test)
        y_test_pred = model.predict(X_test)
        features_val = model.predict_proba(X_val)
        y_val_pred = model.predict(X_val)

    logger.info(f"Feature extraction complete for model type: {model_type}")
    results = []
    uncalibrated_results = compute_uncalibrated_metrics(features_test, y_test_pred, y_test, train_size, val_size, test_size)
    results.append(uncalibrated_results)

    calibrations = {
    "kdtree": {"library": "kdtree", "mode": None},
    "isotonic": IsotonicCalibrator(),
    "platt": PlattCalibrator(),
    "temperature": TemperatureScalingCalibrator(),
    "trust_score_filtered": {"method": "trust_score", "use_filtering": True},
    "trust_score_unfiltered": {"method": "trust_score", "use_filtering": False},
    "faiss_exact": {"library": "faiss", "mode": "exact"},
    "faiss_exact_binned": {"library": "faiss", "mode": "exact", "binned": True},
    "knn": {"library": "knn", "mode": None},
    "knn_binned": {"library": "knn", "mode": None, "binned": True},
    # Add the new calibrators:
    "sbc": SBCCalibrator(bins=15),  # Using SBCCalibrator with 15 bins
    "hb": HBCalibrator(bins=50),    # Using HBCalibrator with 50 bins 
    "bbq": BBQCalibrator(bins=50),  # Using BBQCalibrator with 50 bins
    "beta": BetaCalibrator(bins=50), # Using BetaCalibrator with 50 bins
    "ets": EnsembleTSCalibrator(temperature=1.0),
    "separation": {"library": "separation", "mode": None},  # Add separation
    }


    results, existing_results = perform_calibrations(calibrations, model, X_train, y_train, X_val, y_val, X_test, y_test, features_val, features_test, y_test_pred, technique_dirs, trust_alpha, metric, compression)
    save_results(results, existing_results, all_results_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run geometric calibration experiments.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--random_state", type=int, required=True, help="Random state for reproducibility")
    parser.add_argument("--model_type", type=str, default="cnn", help="Type of model to use (cnn, densenet40)")
    parser.add_argument("--metric", type=str, default="L2", help="Distance metric to use (L1, L2, Linf, cosine)")
    parser.add_argument("--transformed", action="store_true", help="Apply transformations to the test dataset")
    parser.add_argument("--compression_types", type=str, nargs='+', default=None, 
                        help="Compression techniques to use (e.g., Avgpool, Maxpool, PCA)")
    parser.add_argument("--compression_params", type=int, nargs='+', default=[2],
                        help="Parameters for each compression technique")
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        random_state=args.random_state,
        model_type=args.model_type,
        metric=args.metric,
        transformed=args.transformed,
        compression_types=args.compression_types,
        compression_params=args.compression_params
    )