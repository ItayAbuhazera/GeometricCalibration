import bisect
import concurrent
from itertools import repeat
# import scann
# from annoy import AnnoyIndex
# import nmslib
# import hnswlib
import torch
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from math import sqrt

import faiss
import numpy as np
from scipy.optimize import optimize, minimize
import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import scipy.stats
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
import time
import logging
from utils.logging_config import setup_logging

# Configure logging using the singleton configuration
setup_logging()
logger = logging.getLogger(__name__)


# Data class that contains methods to compute stability and separation metrics
class Data:
    """
    Class to hold dataset splits and compute stability and separation metrics.

    Attributes:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Test data.
        X_val (np.ndarray): Validation data.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        y_val (np.ndarray): Validation labels.
        num_labels (int): Number of labels/classes in the dataset.
        isRGB (bool): Indicates if the data is RGB (3 channels) or not (1 channel).
    """

    def __init__(self, X_train, X_test, X_val, y_train, y_test, y_val, num_labels, isRGB=False):
        """
        Initialize the Data object with training, test, and validation data.

        Parameters:
            X_train (np.ndarray): Training data.
            X_test (np.ndarray): Test data.
            X_val (np.ndarray): Validation data.
            y_train (np.ndarray): Training labels.
            y_test (np.ndarray): Test labels.
            y_val (np.ndarray): Validation labels.
            num_labels (int): Number of labels/classes.
            isRGB (bool): Whether the data contains RGB images.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.num_labels = num_labels
        self.isRGB = isRGB
        self.channels = 3 if isRGB else 1
        logger.info(f"Data initialized with {self.num_labels} classes. isRGB: {self.isRGB}")

    def compute_stab(self, whom, y_pred, metric='minkowski'):
        """
        Compute stability for the given data split ('test' or 'val') using a specific metric.

        Parameters:
            whom (str): Dataset split to compute stability for ('test' or 'val').
            y_pred (np.ndarray): Predicted labels.
            metric (str): Distance metric to use ('minkowski' by default).

        Returns:
            np.ndarray: Stability values for the dataset split.
        """
        logger.debug(f"Computing stability for {whom} set using {metric} metric.")
        self._validate_whom(whom)
        if whom == 'test':
            result = stability_calc_knn(self.X_train, self.X_test, self.y_train, y_pred, self.num_labels, metric)
        elif whom == 'val':
            result = stability_calc_knn(self.X_train, self.X_val, self.y_train, y_pred, self.num_labels, metric)
        else:
            logger.error(f"Invalid 'whom' parameter: {whom}. Must be 'test' or 'val'.")
            raise ValueError("Invalid value for 'whom'")
        logger.info(f"Stability computation for {whom} set completed.")
        return result

    def compute_stab_vectored(self, whom, y_pred):
        """
        Compute vectored stability for the given data split ('test' or 'val').

        Parameters:
            whom (str): Dataset split to compute vectored stability for ('test' or 'val').
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Vectored stability values for the dataset split.
        """
        logger.debug(f"Computing vectored stability for {whom} set.")
        self._validate_whom(whom)
        if whom == 'test':
            result = stab_calc_vector(self.X_train, self.X_test, self.y_train, y_pred, self.num_labels)
        elif whom == 'val':
            result = stab_calc_vector(self.X_train, self.X_val, self.y_train, y_pred, self.num_labels)
        else:
            logger.error(f"Invalid 'whom' parameter: {whom}. Must be 'test' or 'val'.")
            raise ValueError("Invalid value for 'whom'")
        logger.info(f"Vectored stability computation for {whom} set completed.")
        return result

    def compute_sep(self, whom, y_pred):
        """
        Compute separation for the given data split ('test' or 'val').

        Parameters:
            whom (str): Dataset split to compute separation for ('test' or 'val').
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Separation values for the dataset split.
        """
        logger.debug(f"Computing separation for {whom} set.")
        self._validate_whom(whom)
        if whom == 'test':
            result = sep_calc(self.X_train, self.X_test, self.y_train, y_pred)
        elif whom == 'val':
            result = sep_calc(self.X_train, self.X_val, self.y_train, y_pred)
        else:
            logger.error(f"Invalid 'whom' parameter: {whom}. Must be 'test' or 'val'.")
            raise ValueError("Invalid value for 'whom'")
        logger.info(f"Separation computation for {whom} set completed.")
        return result

    def get_channels(self):
        """
        Get the number of channels in the data (3 if RGB, 1 otherwise).

        Returns:
            int: Number of channels.
        """
        channels = 3 if self.isRGB else 1
        logger.debug(f"Number of channels: {channels}")
        return channels

    def get_pixels(self):
        """
        Compute the number of pixels in each image by considering the number of channels.

        Returns:
            int: Number of pixels per image (width or height, assuming square images).
        """
        num_of_flatten_pixels = self.X_train.shape[1] / self.channels
        pixels = int(np.sqrt(num_of_flatten_pixels))
        logger.debug(f"Number of pixels per image: {pixels}")
        return pixels

    def _validate_whom(self, whom):
        if whom not in ['test', 'val']:
            logger.error(f"Invalid 'whom' parameter: {whom}. Must be 'test' or 'val'.")
            raise ValueError("Invalid value for 'whom'")


def get_uniform_mass_bins(probs, n_bins):
    """
    Optimized method to create uniform mass bins using percentiles.

    Parameters:
        probs (np.ndarray): Probability values.
        n_bins (int): Number of bins to create.

    Returns:
        np.ndarray: The bin edges for uniform mass bins.
    """
    logger.debug(f"Calculating uniform mass bins for {n_bins} bins.")
    assert probs.size >= n_bins, "Fewer points than bins"

    bin_edges = np.percentile(probs, np.linspace(0, 100, n_bins + 1)[:-1])
    bin_edges[-1] = np.inf  # Ensure final bin is open-ended

    logger.info("Uniform mass bins calculated successfully.")
    return bin_edges


def bin_points(scores, bin_edges):
    """
    Optimized function to bin the points using np.digitize.

    Parameters:
        scores (np.ndarray): Scores to bin.
        bin_edges (np.ndarray): Edges defining the binning intervals.

    Returns:
        np.ndarray: Binned points.
    """
    logger.debug("Binning points based on provided bin edges.")
    assert bin_edges is not None, "Bins have not been defined"

    binned = np.digitize(scores, bin_edges)

    logger.info("Binning completed successfully.")
    return binned


def nudge(matrix, delta):
    """
    Add small random noise to a matrix and normalize it.

    Parameters:
        matrix (np.ndarray): Matrix to nudge.
        delta (float): Amount of noise to add.

    Returns:
        np.ndarray: Nudged matrix.
    """
    logger.debug(f"Nudging matrix with delta {delta}.")
    noise = np.random.uniform(low=0, high=delta, size=matrix.shape)
    return (matrix + noise) / (1 + delta)


def fit_isotonic(xdata, ydata):
    """Fit isotonic regression."""
    logger.debug("Fitting isotonic regression")
    return IsotonicRegression(out_of_bounds="clip").fit(xdata[:, None], ydata)


def fit_linear(xdata, ydata):
    """Fit linear regression."""
    logger.debug("Fitting linear regression")
    model = LinearRegression()
    model.fit(xdata[:, None], ydata)
    return model


def fit_lasso(xdata, ydata):
    """Fit LASSO regression."""
    logger.debug("Fitting LASSO regression")
    model = Lasso()
    model.fit(xdata[:, None], ydata)
    return model


def fit_elastic_net(xdata, ydata):
    """Fit ElasticNet regression."""
    logger.debug("Fitting ElasticNet regression")
    model = ElasticNet()
    model.fit(xdata[:, None], ydata)
    return model


def fit_sigmoid(xdata, ydata, p0):
    """Fit sigmoid curve."""
    logger.debug("Fitting sigmoid function")
    popt, _ = curve_fit(sigmoid_func, xdata, ydata, p0, maxfev=1000000)
    return popt


def fitting_function(stability, y_true, regression_type='isotonic', plot=False):
    """
    Fit a regression model to the stability data (isotonic by default).

    Parameters:
        stability (np.ndarray): Metric for calculation.
        y_true (np.ndarray): True/false classification index.
        regression_type (str): Type of regression to use ('isotonic', 'linear', 'lasso', 'elastic_net').
        plot (bool): Whether to plot the results.

    Returns:
        list: [regression_model, popt_stab_sigmoid].
    """
    logger.info(f"Starting fitting function using {regression_type} regression")

    try:
        # Calculate accuracy for stability values
        s_acc_stab, _, _ = calc_acc(stability, y_true)
        xdata_stab = np.array(list(s_acc_stab.keys()))  # stability
        ydata_stab = np.array(list(s_acc_stab.values()))  # accuracy

        # Dictionary mapping regression type to fitting functions
        regression_methods = {
            'isotonic': fit_isotonic,
            'linear': fit_linear,
            'lasso': fit_lasso,
            'elastic_net': fit_elastic_net,
        }

        # Ensure the specified regression type exists
        if regression_type not in regression_methods:
            logger.error(f"Invalid regression type: {regression_type}")
            raise ValueError(f"Unsupported regression type: {regression_type}")

        # Call the appropriate regression fitting function
        regression_model = regression_methods[regression_type](xdata_stab, ydata_stab)

        # Sigmoid fitting
        p0 = [max(ydata_stab), min(ydata_stab)]
        popt_stab_sigmoid = fit_sigmoid(xdata_stab, ydata_stab, p0)

        logger.info("Fitting function completed successfully")

        return [regression_model, popt_stab_sigmoid]
    except Exception as e:
        logger.error(f"Error occurred during fitting function: {e}")
        raise


def sigmoid_func(x, x0, k):
    """
    Sigmoid function used for curve fitting.

    Parameters:
        x (np.ndarray): Input values.
        x0 (float): Curve midpoint.
        k (float): Steepness of the curve.

    Returns:
        np.ndarray: Sigmoid function values.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def calc_acc(stability, y_true):
    """
    Calculate accuracy for each unique stability value.

    Parameters:
        stability (np.ndarray): Stability calculations.
        y_true (np.ndarray): True binary labels (0/1).

    Returns:
        tuple: (s_acc, s_true, s_all), where:
            s_acc (dict): Accuracy per stability value.
            s_true (dict): Number of true samples per stability value.
            s_all (dict): Total number of samples per stability value.
    """
    logger.info("Calculating accuracy for stability values")

    try:
        stab_values, reps = np.unique(stability, return_counts=True)
        s_true = {stab: 0 for stab in stab_values}
        s_all = dict(zip(stab_values, reps))

        for stab, true_val in zip(stability, y_true):
            s_true[stab] += int(true_val)

        s_acc = {stab: s_true[stab] / s_all[stab] for stab in s_all}

        logger.debug("Accuracy calculation completed successfully")
        return s_acc, s_true, s_all
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        raise


def stability_calc_knn(trainX, testX, train_y, test_y_pred, num_labels, metric='minkowski'):
    """
    Calculate stability using k-Nearest Neighbors (kNN) for a given dataset.

    Parameters:
        trainX (np.ndarray): Training data.
        testX (np.ndarray): Test data.
        train_y (np.ndarray): Training labels.
        test_y_pred (np.ndarray): Predicted labels for the test set.
        num_labels (int): Number of labels/classes.
        metric (str): Distance metric to use for nearest neighbors (default: 'minkowski').

    Returns:
        np.ndarray: Stability values.
    """
    logger.info("Calculating stability for the test set using kNN.")

    def fit_neighbors(label, include_self=True):
        idx = np.where(train_y == label) if include_self else np.where(train_y != label)
        return NearestNeighbors(n_neighbors=1, metric=metric).fit(trainX[idx])

    same_nbrs = [fit_neighbors(i, include_self=True) for i in range(num_labels)]
    other_nbrs = [fit_neighbors(i, include_self=False) for i in range(num_labels)]

    stability = np.zeros(testX.shape[0])

    for i, (x, pred_label) in enumerate(zip(testX, test_y_pred)):
        dist1, _ = same_nbrs[pred_label].kneighbors([x])
        dist2, _ = other_nbrs[pred_label].kneighbors([x])
        stability[i] = (dist2 - dist1) / 2

    logger.info("Stability calculation using kNN completed successfully.")
    return stability



class StabilitySpace:
    """
    Class to compute stability and geometric values for the input X using various similarity search libraries.
    """

    def __init__(self, X_train, y_train, compression=None, library='faiss', metric='l2', num_labels=None):
        """
        Initialize the stability space by compressing the input data (optional) and setting up nearest neighbor models.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing StabilitySpace with {library} library and {metric} metric.")

        self.metric = metric
        self.library = library
        self.num_labels = num_labels or len(set(y_train))
        self.compression = compression

        if self.compression:
            X_train, y_train = self.compression(X_train, y_train)

        self.X_train = X_train
        self.y_train = y_train

        self.logger.info(f"StabilitySpace initialized with {X_train.shape[0]} samples and {self.num_labels} classes.")

        # Initialize nearest neighbor indices for each class
        self.same_nbrs = {}
        self.other_nbrs = {}
        self._build_neighbors_indices()
        # Initialize the stability_calculators dictionary
        self.stability_calculators = {
            'faiss': self._stability_faiss_knn,
            'knn': self._stability_faiss_knn,  # Reuse the same function for KNN
            # Other libraries like 'annoy', 'nmslib', etc. can be added here if needed
        }        

    def _build_neighbors_indices(self):
        """
        Build nearest-neighbor indices for same and other classes.
        """
        self.logger.info("Starting to build nearest-neighbor indices for each class.")
    
        # Reshape X_train if it's multi-dimensional (for example, if it's an image dataset)
        if len(self.X_train.shape) > 2:
            self.logger.debug("Reshaping X_train for FAISS/KNN compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).astype('float32')
    
        dim = self.X_train.shape[1]  # Now dim should be the number of features per sample
        self.logger.info(f"Number of features per sample: {dim}")
    
        # Build separate FAISS indices for same and other-class samples
        for label in range(self.num_labels):
            self.logger.info(f"Processing label {label}/{self.num_labels - 1}.")
    
            # Get indices of samples with the same and different labels
            idx_same = np.where(self.y_train == label)[0]
            idx_other = np.where(self.y_train != label)[0]
    
            if self.library == 'faiss':
                self.logger.info(f"Using FAISS for label {label}.")
    
                # Build FAISS index for same-class data
                try:
                    self.same_nbrs[label] = faiss.IndexFlatL2(dim)
                    self.same_nbrs[label].add(self.X_train[idx_same].astype('float32'))
                    self.logger.info(f"FAISS index for same-class built successfully for label {label}.")
                except Exception as e:
                    self.logger.error(f"Error building FAISS index for same-class for label {label}: {e}")
    
                # Build FAISS index for other-class data
                try:
                    self.other_nbrs[label] = faiss.IndexFlatL2(dim)
                    self.other_nbrs[label].add(self.X_train[idx_other].astype('float32'))
                    self.logger.info(f"FAISS index for other-class built successfully for label {label}.")
                except Exception as e:
                    self.logger.error(f"Error building FAISS index for other-class for label {label}: {e}")
    
    def _stability_faiss_knn(self, trainX, testX, train_y, test_y_pred, num_labels, metric='l2'):
        """
        Calculates the stability of the test set using FAISS or KNN (depending on the prebuilt indices).
        """
        logger.info(f"Starting stability calculation using {self.library} with metric: {metric}.")
        predicted_labels = np.argmax(test_y_pred, axis=1)
        stability = np.zeros(len(testX))

        for i in tqdm(range(testX.shape[0]), desc="Calculating Stability", unit="sample"):
            x = testX[i].reshape(1, -1)  # Flatten the test instance
            pred_label = int(predicted_labels[i])

            if self.library == 'faiss':
                _, dist_same = self.same_nbrs[pred_label].search(x, 1)  # Nearest in same class
                _, dist_other = self.other_nbrs[pred_label].search(x, 1)  # Nearest in other classes

            stability[i] = (dist_other[0][0] - dist_same[0][0]) / 2

        logger.info("Completed stability calculation.")
        return stability

    def calc_stab(self, X_test, y_test_pred):
        """
        Calculate stability for the test set.
        """
        if self.compression:
            X_test, _ = self.compression(X_test, None, train=False)

        self.logger.info(f"Calculating stability using {self.library} with metric {self.metric}.")
        stability_calculator = self.stability_calculators.get(self.library)

        if stability_calculator is not None:
            return stability_calculator(self.X_train, X_test, self.y_train, y_test_pred, self.num_labels, metric=self.metric)
        else:
            self.logger.error(f"Unsupported library: {self.library}")
            raise ValueError(f"Unsupported library: {self.library}")

def calc_balanced_acc(stability, y_true, y_pred):
    '''
    Returns the dicts of description of stability/separation (balanced accuracy, #num of True samples, #num of samples)

    Parameters:
        stability (list of floats or ndarray): Stability list of calculations
        y_true (list): Actual class labels (binary in this case)
        y_pred (list): Predicted class labels (binary in this case)

    Returns:
        s_bal_acc (dict): {key = normalized unique stability, value = balanced accuracy for that stability}
        s_true (dict): {key = normalized unique stability, value = amount of True samples per this stability for each class}
        s_all (dict): {key = normalized unique stability, value = amount of instances exist for that stability for each class}
    '''
    logger.info("Starting the calculation of balanced accuracy.")
    # Get unique stability values and their frequencies
    stab_values, _ = np.unique(stability, return_counts=True)
    logger.info(f"Unique stability values: {stab_values}")

    # Create dictionaries to count True positives and total samples per class for each stability value
    s_true = {stab: np.zeros(2) for stab in stab_values}  # True positives per class
    s_all = {stab: np.zeros(2) for stab in stab_values}  # Total samples per class
    
    # Count true positives and total samples
    for stab, true, pred in zip(stability, y_true, y_pred):
        class_index = int(true)  # Assuming y_true is either 0 or 1
        s_all[stab][class_index] += 1
        if true == pred:
            s_true[stab][class_index] += 1

    logger.info("Completed counting true positives and total samples for each stability value.")

    # Calculate balanced accuracy for each stability
    s_bal_acc = {}
    i = 0
    for stab in stab_values:
        if np.sum(s_all[stab]) == 0:  # Avoid division by zero
            s_bal_acc[stab] = 0
        else:
            # Balanced accuracy: average of recall for each class
            class_recalls = [s_true[stab][i] / s_all[stab][i] if s_all[stab][i] != 0 else 0 for i in range(2)]
            s_bal_acc[stab] = np.mean(class_recalls)
        if i % 400 == 0:
            logger.info(f"Stability {stab}: Balanced Accuracy = {s_bal_acc[stab]}")
        i += 1

    return s_bal_acc, s_true, s_all
# def calc_balanced_acc(stability, y_true, y_pred, num_classes, round_digits=2):
#     """
#     Returns the balanced accuracy for each rounded stability value in multi-class classification.

#     Parameters:
#         stability (np.ndarray): Stability values.
#         y_true (np.ndarray): True class labels.
#         y_pred (np.ndarray): Predicted class labels.
#         num_classes (int): Number of classes.
#         round_digits (int): Number of decimal places to round the stability values.

#     Returns:
#         dict: {rounded_stability: mean balanced accuracy}
#     """
#     # Round the stability values to the desired precision
#     rounded_stability = np.round(stability, decimals=round_digits)
#     logger.info(f"Rounded stability values (first 10): {rounded_stability[:10]}")

#     # Get the unique rounded stability values
#     stab_values, _ = np.unique(rounded_stability, return_counts=True)
#     logger.info(f"Unique rounded stability values: {stab_values}")

#     # Initialize dictionaries for counts of true positives and total samples per class
#     s_true = {stab: np.zeros(num_classes) for stab in stab_values}
#     s_all = {stab: np.zeros(num_classes) for stab in stab_values}

#     # Count true positives and total samples for each rounded stability value
#     for stab, true_label, pred_label in zip(rounded_stability, y_true, y_pred):
#         # Count the total samples for the true class
#         s_all[stab][true_label] += 1
#         # If the prediction is correct, count it as a true positive for that class
#         if true_label == pred_label:
#             s_true[stab][true_label] += 1

#     logger.info("Counts of true positives and total samples per stability value:")
#     i = 0
#     for stab in stab_values:
#         if i%500 == 0:
#             logger.info(f"Stability value: {stab} | True counts: {s_true[stab]} | Total counts: {s_all[stab]}")
#         i += 1

#     # Calculate balanced accuracy
#     s_bal_acc = {}
#     i = 0
#     for stab in stab_values:
#         # Recall (per class): True Positives / All Samples (per class)
#         recalls = [s_true[stab][cls] / s_all[stab][cls] if s_all[stab][cls] > 0 else 0 for cls in range(num_classes)]
#         # Balanced accuracy is the average of recall values across all classes
#         s_bal_acc[stab] = np.mean(recalls)
#         if i%500 == 0:
#             logger.info(f"Stability value: {stab} | Recalls: {recalls} | Balanced Accuracy: {s_bal_acc[stab]}")
#         i += 1
#     return s_bal_acc

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean confidence interval for a given data set.

    Parameters:
        data (list or np.ndarray): Input data.
        confidence (float): Confidence level (default is 0.95).

    Returns:
        tuple: (mean, lower bound, upper bound)
    """
    logger.info("Calculating mean confidence interval")

    try:
        a = np.array(data)
        m = np.mean(a)
        se = scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a) - 1)

        return m, m - h, m + h
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {e}")
        raise


# Utility functions for optimization (from ensemble_ts.py)
def mse_t(t, *args):
    """
    Mean Squared Error loss function for temperature scaling.

    Parameters:
        t (float): Temperature value to scale logits.
        *args: Logits and labels to compute MSE.

    Returns:
        float: Mean squared error between scaled logits and labels.
    """
    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.exp(logit) / n[:, None]
    mse = np.mean((p - label) ** 2)
    logger.debug(f"MSE computed for temperature {t}: {mse}")
    return mse


def ll_t(t, *args):
    """
    Cross-entropy loss function for temperature scaling.

    Parameters:
        t (float): Temperature value to scale logits.
        *args: Logits and labels to compute cross-entropy.

    Returns:
        float: Cross-entropy loss between scaled logits and labels.
    """
    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.clip(np.exp(logit) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    logger.debug(f"Cross-entropy loss computed for temperature {t}: {ce}")
    return ce


def temperature_scaling(logit, label, loss='ce'):
    """
    Perform temperature scaling for logits using the specified loss function.

    Parameters:
        logit (np.ndarray): Logits from the model (pre-softmax).
        label (np.ndarray): True labels (one-hot encoded or probability distribution).
        loss (str): Loss function to use ('ce' for cross-entropy, 'mse' for mean squared error).

    Returns:
        float: Optimized temperature scaling value.
    """
    logger.info(f"Performing temperature scaling using {loss} loss.")

    loss_fn = ll_t if loss == 'ce' else mse_t if loss == 'mse' else None
    if loss_fn is None:
        logger.error(f"Invalid loss function: {loss}")
        raise ValueError(f"Loss {loss} not supported.")

    t = minimize(loss_fn, 1.0, args=(logit, label), method='L-BFGS-B', bounds=((0.05, 5.0),), tol=1e-12)

    logger.info(f"Temperature scaling completed. Optimal temperature: {t.x[0]}")
    return t.x


# Function to split and save data into HDF5 format
# def split_and_save_range_hdf5(train_X_original, test_X_original, train_y_original, test_y_original, split_range, dataset_name):
#     """
#     Splits the data into a range of chunks and saves them in HDF5 format.
#
#     Parameters:
#         train_X_original (np.ndarray): Original training features.
#         test_X_original (np.ndarray): Original test features.
#         train_y_original (np.ndarray): Original training labels.
#         test_y_original (np.ndarray): Original test labels.
#         split_range (range): Range of splits for data shuffling.
#         dataset_name (str): Name of the dataset for consistency in loading.
#
#     Returns:
#         None
#     """
#     logger.info("Starting data splitting and saving to HDF5 format.")
#
#     # Flatten the image data (from 28x28 to 784)
#     pixels = train_X_original.shape[1] * train_X_original.shape[2]
#
#     trainX = train_X_original.reshape(len(train_X_original), pixels).astype(np.float64)
#     testX = test_X_original.reshape(len(test_X_original), pixels).astype(np.float64)
#
#     # Ensure labels are simple 1D arrays
#     train_y_original = np.array(train_y_original).astype(np.int64).reshape(-1)
#     test_y_original = np.array(test_y_original).astype(np.int64).reshape(-1)
#
#     for i in split_range:
#         # Concatenate data for train/test split
#         data = np.concatenate((trainX, testX), axis=0)
#         y = np.concatenate((train_y_original, test_y_original), axis=0)
#
#         # Split into train/test/validation sets
#         X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=i)
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
#
#         directory = f'./{dataset_name}/{i}/data/'
#         os.makedirs(directory, exist_ok=True)
#
#         # Convert labels to simple 1D arrays before saving
#         y_train = np.array(y_train).reshape(-1)
#         y_test = np.array(y_test).reshape(-1)
#         y_val = np.array(y_val).reshape(-1)
#
#         # Save the datasets into HDF5 file
#         with h5py.File(os.path.join(directory, 'dataset.h5'), 'w') as hf:
#             hf.create_dataset('X_train', data=X_train)
#             hf.create_dataset('X_test', data=X_test)
#             hf.create_dataset('X_val', data=X_val)
#             hf.create_dataset('y_train', data=y_train)
#             hf.create_dataset('y_test', data[y_test)
#             hf.create_dataset('y_val', data=y_val)
#
#         logger.info(f"Data saved successfully for shuffle {i}.")


# Function to load data from HDF5 format
def load_data_hdf5(dataset_name, shuffle_num):
    """
    Loads the dataset from HDF5 format for a specific shuffle.

    Parameters:
        dataset_name (str): Name of the dataset.
        shuffle_num (int): Shuffle number to load the dataset.

    Returns:
        Data: A Data object containing training, test, and validation data splits.
    """
    logger.info(f"Loading dataset {dataset_name} for shuffle {shuffle_num}.")

    try:
        VARS = json.load(open('../SLURM/VARS.json'))
    except Exception:
        VARS = json.load(open('./SLURM/VARS.json'))

    NUM_LABELS = VARS['NUM_LABELS']
    data_dir = f'./{dataset_name}/{shuffle_num}/data/'
    file_path = os.path.join(data_dir, 'dataset.h5')

    # Load data from HDF5 file
    with h5py.File(file_path, 'r') as hf:
        X_train = hf['X_train'][:]
        X_test = hf['X_test'][:]
        X_val = hf['X_val'][:]
        y_train = hf['y_train'][:]
        y_test = hf['y_test'][:]
        y_val = hf['y_val'][:]

    isRGB = "RGB" in dataset_name
    data = Data(X_train, X_test, X_val, y_train, y_test, y_val, NUM_LABELS[dataset_name], isRGB)

    logger.info(f"Dataset {dataset_name} loaded successfully for shuffle {shuffle_num}.")
    return data


########## stability vs separation ####################

def stab_calc_vector(X_train, X_test, y_train, y_pred_test, num_labels):
    """
    Calculates the stability of the test set using vectorization.

    Parameters:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training labels.
        y_pred_test (np.ndarray): Predicted labels for test set.
        num_labels (int): Number of distinct labels/classes.

    Returns:
        np.ndarray: Vector of stability values with None in predicted label places.
    """
    logger.info("Calculating vectored stability for the test set.")

    # Initialize stability array
    stabs = np.zeros((len(X_test), num_labels))

    # Precompute 1NN trees for each class
    same_nbrs = [NearestNeighbors(n_neighbors=1).fit(X_train[y_train == i]) for i in range(num_labels)]

    for i, x in enumerate(tqdm(X_test)):
        pred = y_pred_test[i]
        dist1, _ = same_nbrs[pred].kneighbors([x])

        for label in range(num_labels):
            if label == pred:
                stabs[i, label] = None
            else:
                dist2, _ = same_nbrs[label].kneighbors([x])
                stabs[i, label] = (dist2 - dist1) / 2

    logger.info("Stability vector computation completed.")
    return stabs


# Universal distance metric mapping
UNIVERSAL_METRICS = {
    'l2': {'faiss': 'l2', 'scann': 'squared_l2', 'annoy': 'euclidean', 'nmslib': 'l2', 'hnsw': 'l2'},
    'cosine': {'faiss': 'cosine', 'scann': 'dot_product', 'annoy': 'angular', 'nmslib': 'cosinesimil',
               'hnsw': 'cosine'},
    'inner_product': {'faiss': 'inner_product', 'scann': 'dot_product', 'annoy': 'dot', 'nmslib': 'ip', 'hnsw': None}
}


def get_library_metric(library, metric):
    """
    Get the corresponding metric name for a specific library.

    Parameters:
        library (str): The library name (e.g., 'faiss', 'scann').
        metric (str): The universal metric name (e.g., 'l2', 'cosine').

    Returns:
        str: The metric name used in the given library.
    """
    if metric not in UNIVERSAL_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    library_metric = UNIVERSAL_METRICS[metric].get(library)
    if library_metric is None:
        raise ValueError(f"Unsupported metric for {library}: {metric}")
    return library_metric


def stability_calc_faiss(trainX, testX, train_y, test_y_pred, num_labels, metric='l2'):
    """
    Calculates the stability of the test set using FAISS (Facebook AI Similarity Search),
    mimicking the structure of the original stability calculation method.

    Parameters:
        trainX (np.ndarray): Training features.
        testX (np.ndarray): Test features.
        train_y (np.ndarray): Training labels.
        test_y_pred (np.ndarray): Predicted probabilities for the test set.
        num_labels (int): Number of distinct labels.
        metric (str): Distance metric to use ('l2', 'inner_product', etc.).

    Returns:
        np.ndarray: Stability values for each test instance.
    """
    logger.info(f"Starting stability calculation using FAISS with metric: {metric}.")

    try:
        # Reshape and flatten the data to be compatible with FAISS
        if len(trainX.shape) > 2:
            trainX = trainX.reshape(trainX.shape[0], -1).astype('float32')
            testX = testX.reshape(testX.shape[0], -1).astype('float32')

        dim = trainX.shape[1]
        faiss_metric = get_library_metric('faiss', metric)

        # Choose the appropriate FAISS index for nearest neighbors
        if faiss_metric == 'l2':
            faiss_index = faiss.IndexFlatL2(dim)
        elif faiss_metric == 'inner_product':
            faiss_index = faiss.IndexFlatIP(dim)
        elif faiss_metric == 'cosine':
            faiss.normalize_L2(trainX)
            faiss.normalize_L2(testX)
            faiss_index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported FAISS metric: {faiss_metric}")

        # Add training data to FAISS index
        faiss_index.add(trainX)

        # Search for nearest neighbors in the test set
        _, all_distances = faiss_index.search(testX, 1)

        # Convert predicted probabilities into class labels using argmax
        predicted_labels = np.argmax(test_y_pred, axis=1)

        stability = np.zeros(len(testX))

        # For each test instance, calculate stability
        for i in tqdm(range(testX.shape[0]), desc="Calculating Stability", unit="sample"):
            x = testX[i].reshape(1, -1)  # Flatten the test instance
            pred_label = int(predicted_labels[i])  # Convert predicted probabilities to class label

            # Find the nearest neighbor in the "same" class and "other" classes
            _, dist_same = faiss_index.search(trainX[train_y == pred_label], 1)  # Nearest in same class
            _, dist_other = faiss_index.search(trainX[train_y != pred_label], 1)  # Nearest in other classes

            # Calculate stability based on the difference in distances
            stability[i] = (dist_other[0] - dist_same[0]) / 2

        logger.info("Completed stability calculation using FAISS.")
        return stability

    except Exception as e:
        logger.error(f"Error in FAISS stability calculation: {e}")
        raise

# def stability_calc_scann(X_train, X_test, y_train, y_pred_test, num_labels, metric='dot_product'):
#     """
#     Calculates the stability of the test set using ScaNN.
#
#     Parameters:
#         X_train (np.ndarray): Training features.
#         X_test (np.ndarray): Test features.
#         y_train (np.ndarray): Training labels.
#         y_pred_test (np.ndarray): Predicted labels for test set.
#         num_labels (int): Number of distinct labels.
#         metric (str): Distance metric to use ('dot_product', 'squared_l2').
#
#     Returns:
#         np.ndarray: Stability values for each test instance.
#     """
#     logger.info(f"Starting stability calculation using ScaNN with metric: {metric}.")
#
#     try:
#         scann_metric = get_library_metric('scann', metric)
#         searchers = [scann.scann_ops_pybind.builder(X_train[y_train == i], 1, scann_metric).build() for i in
#                      range(num_labels)]
#         stability = np.zeros((len(X_test), num_labels))
#
#         for i, x in enumerate(tqdm(X_test)):
#             pred = y_pred_test[i]
#             dist1, _ = searchers[pred].search(np.array([x]))
#
#             for label in range(num_labels):
#                 if label == pred:
#                     stability[i, label] = None
#                 else:
#                     dist2, _ = searchers[label].search(np.array([x]))
#                     stability[i, label] = (dist2 - dist1) / 2
#
#         logger.info("Completed stability calculation using ScaNN.")
#         return stability
#     except Exception as e:
#         logger.error(f"Error in ScaNN stability calculation: {e}")
#         raise


# def stability_calc_annoy(X_train, X_test, y_train, y_pred_test, num_labels, metric='euclidean'):
#     """
#     Calculates the stability of the test set using Annoy.

#     Parameters:
#         X_train (np.ndarray): Training features.
#         X_test (np.ndarray): Test features.
#         y_train (np.ndarray): Training labels.
#         y_pred_test (np.ndarray): Predicted labels for test set.
#         num_labels (int): Number of distinct labels.
#         metric (str): Distance metric to use ('euclidean', 'manhattan', 'dot').

#     Returns:
#         np.ndarray: Stability values for each test instance.
#     """
#     logger.info(f"Starting stability calculation using Annoy with metric: {metric}.")

#     try:
#         annoy_metric = get_library_metric('annoy', metric)
#         dim = X_train.shape[1]

#         annoy_trees = []
#         for label in range(num_labels):
#             t = AnnoyIndex(dim, annoy_metric)
#             for i, vec in enumerate(X_train[y_train == label]):
#                 t.add_item(i, vec)
#             t.build(10)
#             annoy_trees.append(t)

#         stability = np.zeros((len(X_test), num_labels))

#         for i, x in enumerate(X_test):
#             pred = y_pred_test[i]
#             dist1 = annoy_trees[pred].get_nns_by_vector(x, 1, include_distances=True)[1][0]

#             for label in range(num_labels):
#                 if label == pred:
#                     stability[i, label] = None
#                 else:
#                     dist2 = annoy_trees[label].get_nns_by_vector(x, 1, include_distances=True)[1][0]
#                     stability[i, label] = (dist2 - dist1) / 2

#         logger.info("Completed stability calculation using Annoy.")
#         return stability
#     except Exception as e:
#         logger.error(f"Error in Annoy stability calculation: {e}")
#         raise


# def stability_calc_nmslib(X_train, X_test, y_train, y_pred_test, num_labels, metric='l2'):
#     """
#     Calculates the stability of the test set using NMSLIB.

#     Parameters:
#         X_train (np.ndarray): Training features.
#         X_test (np.ndarray): Test features.
#         y_train (np.ndarray): Training labels.
#         y_pred_test (np.ndarray): Predicted labels for test set.
#         num_labels (int): Number of distinct labels.
#         metric (str): Distance metric to use ('l2', 'cosine', etc.).

#     Returns:
#         np.ndarray: Stability values for each test instance.
#     """
#     logger.info(f"Starting stability calculation using NMSLIB with metric: {metric}.")

#     try:
#         nmslib_metric = get_library_metric('nmslib', metric)
#         indices = []
#         for i in range(num_labels):
#             idx_same = nmslib.init(method='hnsw', space=nmslib_metric)
#             idx_same.addDataPointBatch(X_train[y_train == i])
#             idx_same.createIndex({'post': 2}, print_progress=True)
#             indices.append(idx_same)

#         stability = np.zeros(len(X_test))

#         for i, x in enumerate(tqdm(X_test)):
#             pred_label = y_pred_test[i]
#             dist1, _ = indices[pred_label].knnQuery(x, k=1)

#             min_other_distances = []
#             for label in range(num_labels):
#                 if label == pred_label:
#                     continue
#                 dist2, _ = indices[label].knnQuery(x, k=1)
#                 min_other_distances.append(dist2[0])

#             stability[i] = (min(min_other_distances) - dist1[0]) / 2

#         logger.info("Completed stability calculation using NMSLIB.")
#         return stability
#     except Exception as e:
#         logger.error(f"Error in NMSLIB stability calculation: {e}")
#         raise


# def stability_calc_hnsw(X_train, X_test, y_train, y_pred_test, num_labels, metric='l2'):
#     """
#     Calculates the stability of the test set using HNSWlib.

#     Parameters:
#         X_train (np.ndarray): Training features.
#         X_test (np.ndarray): Test features.
#         y_train (np.ndarray): Training labels.
#         y_pred_test (np.ndarray): Predicted labels for test set.
#         num_labels (int): Number of distinct labels.
#         metric (str): Distance metric to use ('l2', 'cosine').

#     Returns:
#         np.ndarray: Stability values for each test instance.
#     """
#     logger.info(f"Starting stability calculation using HNSWlib with metric: {metric}.")

#     try:
#         hnsw_metric = get_library_metric('hnsw', metric)
#         dim = X_train.shape[1]
#         indices = []
#         for i in range(num_labels):
#             index = hnswlib.Index(space=hnsw_metric, dim=dim)
#             index.init_index(max_elements=len(X_train[y_train == i]), ef_construction=100, M=16)
#             index.add_items(X_train[y_train == i])
#             indices.append(index)

#         stability = np.zeros(len(X_test))

#         for i, x in enumerate(tqdm(X_test)):
#             pred_label = y_pred_test[i]
#             dist1, _ = indices[pred_label].knn_query(x, k=1)

#             min_other_distances = []
#             for label in range(num_labels):
#                 if label == pred_label:
#                     continue
#                 dist2, _ = indices[label].knn_query(x, k=1)
#                 min_other_distances.append(dist2[0][0])

#             stability[i] = (min(min_other_distances) - dist1[0][0]) / 2

#         logger.info("Completed stability calculation using HNSWlib.")
#         return stability
#     except Exception as e:
#         logger.error(f"Error in HNSWlib stability calculation: {e}")
#         raise


def sep_calc_parallel_sklearn(testX, pred_y, data_dir, norm='L2'):
    """
    Parallel computation of separation using sklearn across multiple processes.
    """
    logger.info("Calculating separation in parallel using sklearn.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        separation = list(executor.map(sep_parallel_sklearn, testX, pred_y, repeat(data_dir), repeat(norm)))
    logger.info("Completed parallel separation calculation using sklearn.")
    return separation


def sep_parallel_sklearn(x, pred, data_dir, norm='L1'):
    """
    Calculate the separation for a single point using sklearn's NearestNeighbors, loading data from HDF5.
    """
    with h5py.File(os.path.join(data_dir, 'dataset.h5'), 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
    return sep_calc_point(x, X_train, y_train, pred, norm)


def sep_calc_parallel(testX, pred_y, data_dir, norm='L2'):
    """
    Parallel computation of separation using custom parallel processing.
    """
    logger.info("Calculating separation in parallel.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        separation = list(executor.map(sep_parallel, testX, pred_y, repeat(data_dir), repeat(norm)))
    logger.info("Completed parallel separation calculation.")
    return separation


def sep_parallel(x, pred, data_dir, norm='L1'):
    """
    Calculate separation for a single test instance using data loaded from HDF5.
    """
    with h5py.File(os.path.join(data_dir, 'dataset.h5'), 'r') as hf:
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
    return sep_calc_point(x, X_train, y_train, pred, norm)


def sep_calc_point(x, trainX, train_y, y, norm='L1'):
    """
    Calculate the separation of a single point from training data.

    Parameters:
        x (np.ndarray): Test instance.
        trainX (np.ndarray): Training instances.
        train_y (np.ndarray): Training labels.
        y (int): Predicted class label for the test instance.
        norm (str): Distance norm to use ('L1', 'L2', 'Linf').

    Returns:
        float: Separation score for the test instance.
    """
    logger.debug(f"Calculating separation for class {y} with norm {norm}.")

    # Map norm to correct value
    norms_map = {'L1': 1, 'L2': 2, 'Linf': np.inf}
    norm_value = norms_map.get(norm)
    if norm_value is None:
        raise ValueError(f"Unsupported norm: {norm}")

    # Calculate distances for same and different classes
    same_class_mask = (train_y == y)
    dist_same = np.linalg.norm(trainX[same_class_mask] - x, ord=norm_value, axis=1)
    dist_other = np.linalg.norm(trainX[~same_class_mask] - x, ord=norm_value, axis=1)

    # Calculate separation
    min_same = np.min(dist_same) if len(dist_same) > 0 else float('inf')
    min_other = np.min(dist_other) if len(dist_other) > 0 else float('inf')
    separation = (min_other - min_same) / 2 if min_same < float('inf') and min_other < float('inf') else float('inf')

    logger.debug(f"Separation for class {y}: {separation}")
    return separation


def sep_calc_point_faiss(x, trainX, train_y, y, norm='L2'):
    """
    Calculate the separation using FAISS for a single point.

    Parameters:
        x (np.ndarray): Test instance.
        trainX (np.ndarray): Training instances.
        train_y (np.ndarray): Training labels.
        y (int): Predicted class label for the test instance.
        norm (str): Distance norm to use (currently only 'L2' is supported).

    Returns:
        float: Separation score for the test instance.
    """
    logger.info("Calculating separation using FAISS.")
    dim = trainX.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(trainX)

    idx1 = np.where(train_y == y)[0]
    _, idx1_faiss = faiss_index.search(np.array([x]), 1)
    dist1 = np.linalg.norm(x - trainX[idx1_faiss[0][0]])

    idx2 = np.where(train_y != y)[0]
    _, idx2_faiss = faiss_index.search(trainX[idx2], 1)
    dist2 = np.linalg.norm(x - trainX[idx2_faiss[0][0]])

    separation = (dist2 - dist1) / 2
    logger.info("Completed separation calculation using FAISS.")
    return separation


def sep_calc(trainX, testX, train_y, pred_y, norm='L1'):
    """
    Calculate the separation for all the examples in test/validation set.

    Parameters:
        trainX (np.ndarray): Training instances.
        testX (np.ndarray): Test/validation instances.
        train_y (np.ndarray): Training labels.
        pred_y (np.ndarray): Predicted labels for the test/validation set.
        norm (str or int): Norm to calculate distance ('L1', 'L2', 'Linf').

    Returns:
        List[float]: Separation values for each instance in the test/validation set.
    """
    logger.info("Starting separation calculation.")
    separation = [sep_calc_point(x, trainX, train_y, pred_y[i], norm) for i, x in enumerate(testX)]
    logger.info("Completed separation calculation.")
    return separation


# def ECE_calc(probs, y_pred, y_real, bins=15):
#     """
#     Calculate the Expected Calibration Error (ECE) for model predictions.

#     Parameters:
#         probs (np.ndarray): Probabilities predicted by the model.
#         y_pred (np.ndarray): Predicted class labels.
#         y_real (np.ndarray): True class labels.
#         bins (int): Number of bins for calibration calculation (default 15).

#     Returns:
#         float: Expected Calibration Error (ECE).
#     """
#     logger.info("Starting ECE calculation.")

#     def gap_calc(lst):
#         if lst == [0]:
#             return 0
#         s_lst = sum(lst[1:])
#         l_lst = len(lst[1:])
#         avg = s_lst / l_lst
#         accuracy = lst[0] / l_lst
#         return abs(avg - accuracy) * l_lst

#     if isinstance(probs, np.ndarray) and len(probs.shape) == 2:
#         probs = [max(i) for i in probs]

#     lin_space = np.linspace(0, 1, bins + 1)
#     ziped = list(zip(probs, y_pred == y_real))
#     ziped.sort(key=lambda x: x[0])

#     b = [[0] for i in range(len(lin_space))]
#     b_num = 0
#     for x in ziped:
#         p = x[0]
#         inserted = False
#         while not inserted:
#             if p == 1:
#                 b[-2].append(p)
#                 inserted = True
#             elif p < lin_space[b_num + 1]:
#                 b[b_num].append(p)
#                 inserted = True
#             else:
#                 b_num += 1
#         if x[1]:
#             if p == 1:
#                 b[-2][0] += 1
#             else:
#                 b[b_num][0] += 1

#     ECE_sum = 0
#     for idx, data in enumerate(b):
#         ECE_sum += gap_calc(data)

#     ECE = ECE_sum / len(y_pred)
#     logger.info(f"ECE calculation completed with result: {ECE}")
#     return ECE


def plot_fitting_function(model_info, n_bins, save=False):
    """
    Plot the fitting function for a model's separation and accuracy data.

    Parameters:
        model_info (object): Object containing model data (e.g., stability values, predictions).
        n_bins (int): Number of bins for the plot.
        save (bool): Whether to save the plot to a file (default: False).
    """
    logger.info("Plotting fitting function.")
    stab_latex = r'$\underline{\mathcal{S}}^{\mathcal{M}}$'
    correct = model_info.y_pred_val == model_info.data.y_val
    popt = fitting_function(model_info.stability_val, correct)

    ylabels = model_info.y_pred_test == model_info.data.y_test
    xlabels = model_info.stability_test
    length = (max(xlabels) - min(xlabels)) / n_bins

    bins_data = [0 for _ in range(n_bins + 1)]
    bins_data_num = [0 for _ in range(n_bins + 1)]
    for i in range(len(xlabels)):
        bins_data[int((xlabels[i] - min(xlabels)) / length)] += ylabels[i]
        bins_data_num[int((xlabels[i] - min(xlabels)) / length)] += 1

    ydata = [[], []]
    xdata = [[], []]
    plot_x = []
    y_data_return = []
    colors = ["r", "b"]
    markers = ['o', "D"]

    for i in range(n_bins + 1):
        if bins_data_num[i] == 0:
            continue
        idx = 0 if bins_data_num[i] < 100 else 1
        ydata[idx].append(bins_data[i] / bins_data_num[i])
        xdata[idx].append(length * i + min(xlabels))
        plot_x.append(length * i + min(xlabels))
        y_data_return.append(bins_data[i] / bins_data_num[i])

    plt.xlabel(f'Fast Separation Score {stab_latex}')
    plt.ylabel("Accuracy on Validation Set")

    for i in range(len(colors)):
        plt.scatter(xdata[i], ydata[i], c=colors[i], marker=markers[i])

    xdata = np.array(plot_x)
    plt.plot(xdata, sigmoid_func(xdata, *popt[1]), color='k')
    plt.plot(xdata, popt[0].predict(xdata.reshape(-1, 1)), color='g')
    plt.legend(["Less than 100 samples", "More than 100 samples", "Sigmoid fitting", "Isotonic regression"])

    if save:
        plt.savefig('plot.pdf')
    plt.show()
    logger.info("Plotting fitting function completed.")


def normalize_dataset(data):
    """
    Normalize the dataset (image data) by subtracting mean and dividing by standard deviation.

    Parameters:
        data (np.ndarray): Image data array.

    Returns:
        Normalize: Torchvision normalization object.
    """
    logger.info("Normalizing dataset.")
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    normalize = transforms.Normalize(mean=mean, std=std)
    logger.info(f"Dataset normalization completed. Mean: {mean}, Std: {std}")
    return normalize


def hot_padding(oneDim, positions, num_labels):
    """
    Apply one-hot encoding with padding to the provided data.

    Parameters:
        oneDim (np.ndarray): 1D array of probabilities.
        positions (np.ndarray): Positions where values should be placed.
        num_labels (int): Number of labels/classes.

    Returns:
        np.ndarray: One-hot encoded and padded array.
    """
    logger.info("Applying hot padding to data.")
    hot_padding_probs = np.zeros((len(oneDim), num_labels))
    for i, pos in enumerate(positions):
        hot_padding_probs[i][pos] = oneDim[i]

    logger.info("Hot padding completed.")
    return hot_padding_probs


def get_bin(s, bins_ranges):
    """
    Get the bin index for a value based on bin ranges.

    Parameters:
        s (float): Value to bin.
        bins_ranges (np.ndarray): The array of bin ranges.

    Returns:
        int: Bin index.
    """
    return bisect.bisect(bins_ranges, s) - 1


def histogramBinning(probs, corrects, num_bins):
    """
    Apply histogram binning to the probability values.

    Parameters:
        probs (np.ndarray): Predicted probabilities.
        corrects (np.ndarray): Correctness of predictions (1 if correct, 0 if incorrect).
        num_bins (int): Number of bins.

    Returns:
        tuple: Mean values per bin, bin ranges, and new ranges.
    """
    logger.info("Applying histogram binning.")
    bins_nums, bins_ranges = np.histogram(probs, bins=num_bins)
    binned_values = [[] for _ in range(num_bins)]

    for prob, value in zip(probs, corrects):
        bin_idx = get_bin(prob, bins_ranges)
        binned_values[bin_idx].append(float(value))

    for idx, bin_data in enumerate(binned_values):
        if not bin_data:
            binned_values[idx] = [np.nan]

    bin_means = [np.mean(values) for values in binned_values]
    for idx, val in enumerate(bin_means):
        if np.isnan(val):
            bin_means[idx] = bin_means[idx - 1]

    new_ranges = [(bins_ranges[i - 1] + bins_ranges[i]) / 2 for i in range(1, len(bins_ranges))]
    logger.info("Histogram binning completed.")
    return bin_means, bins_ranges, new_ranges


######################################  Dataframe formatting functions ######################################

def color_max(s):
    """
    Apply background color to the maximum value in a Pandas DataFrame row.

    Parameters:
        s (pd.Series): DataFrame row.

    Returns:
        list: List of styles for the DataFrame row.
    """
    numbers = np.array([float(i) if isinstance(i, str) and len(i) > 1 else np.inf for i in s])
    is_max = numbers == min(numbers)
    return ['background-color: darkgreen' if v else '' for v in is_max]


def percentage_format(x):
    """
    Format a string to percentage format.

    Parameters:
        x (str): String value to format.

    Returns:
        str: Formatted string.
    """
    if isinstance(x, str) and '±' in x:
        a, b = x.split('±')
        return f'{float(a) * 100:.2f}±{float(b) * 100:.2f}'
    return x


def mean_confidence_interval_str(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for a list of data.

    Parameters:
        data (list): List of numerical values.
        confidence (float): Confidence interval level.

    Returns:
        str: Formatted mean and confidence interval string.
    """
    if isinstance(data, list):
        a = np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return f'{m:.6f}±{h:.6f}'
    return '-'


def non_zero_format(x):
    """
    Format non-zero values for display in a DataFrame.

    Parameters:
        x (str): String containing the value to format.

    Returns:
        str: Formatted string.
    """
    if x != '-':
        a, b = x.split('±')
        a = a[1:] if a.startswith('0.') else a
        b = b[1:] if b.startswith('0.') else b
        return f'{a}±{b}'
    return x


def order_by(indexes, order, second_order):
    """
    Reorder indexes based on the primary and secondary order.

    Parameters:
        indexes (list): List of indexes to reorder.
        order (list): Primary ordering.
        second_order (list): Secondary ordering.

    Returns:
        list: Reordered indexes.
    """
    ans = []
    first_sort = [[item for item in indexes if item.startswith(p)] for p in order]
    for lst in first_sort:
        for p in second_order:
            ans.extend([item for item in lst if item.endswith(p)])
    return ans


class Compression:
    """
    Class used for compressing data using various techniques like AvgPool, MaxPool, PCA, etc.
    It supports applying multiple compression techniques in sequence.
    """

    def __init__(self, compression_types, compression_params):
        """
        Initialize the Compression class.

        Parameters:
            compression_types (str or list): Compression techniques to apply (e.g., "Avgpool", "Maxpool", "PCA", etc.).
            compression_params (int or list): Compression parameters corresponding to each technique.
        """
        self.compression_types = compression_types if isinstance(compression_types, list) else [
            compression_types]
        self.compression_params = compression_params if isinstance(compression_params, list) else [
            compression_params]
        self.pca_model = None

        # Dictionary mapping compression techniques to their functions
        self.compression_methods = {
            'Avgpool': self.avg_pool,
            'Maxpool': self.max_pool,
            'resize': self.resize,
            'PCA': self.pca,
            'randpix': self.randpix,
            'randset': self.randset
        }

    def __call__(self, X_train, y_train, train=True):
        """
        Apply the compression techniques sequentially on the input data.

        Parameters:
            X_train (ndarray): The training data.
            y_train (ndarray): The training labels.
            train (bool): Whether this is training data or test data (affects PCA and randset).

        Returns:
            Tuple: Compressed X_train and y_train.
        """
        pixels = int(sqrt(X_train.shape[1]))
        if not sqrt(X_train.shape[1]).is_integer():
            print("Running without compression, the shape of X needs to be square")
            return X_train, y_train

        X_train = X_train.reshape(len(X_train), pixels, pixels)

        for comp_type, param in zip(self.compression_types, self.compression_params):
            if comp_type in self.compression_methods:
                X_train, y_train = self.compression_methods[comp_type](X_train, y_train, param, train)
            else:
                print(f"No compression method found for {comp_type}")
                return X_train, y_train

        return X_train.reshape(len(X_train), -1), y_train

    def avg_pool(self, X_train, y_train, param, train):
        pool = torch.nn.AvgPool2d(param)
        X_train = pool(torch.tensor(X_train)).reshape(len(X_train), -1).numpy()
        return X_train, y_train

    def max_pool(self, X_train, y_train, param, train):
        pool = torch.nn.MaxPool2d(param)
        X_train = pool(torch.tensor(X_train)).reshape(len(X_train), -1).numpy()
        return X_train, y_train

    def resize(self, X_train, y_train, param, train):
        size = X_train.shape[1] // param
        X_train = tf.image.resize(X_train[..., np.newaxis], [size, size]).numpy().reshape(len(X_train), -1)
        return X_train, y_train

    def pca(self, X_train, y_train, param, train):
        size = X_train.shape[1] // param
        if train:
            self.pca_model = PCA(n_components=size ** 2)
            X_train = self.pca_model.fit_transform(X_train.reshape(len(X_train), -1))
        else:
            X_train = self.pca_model.transform(X_train.reshape(len(X_train), -1))
        return X_train, y_train

    def randpix(self, X_train, y_train, param, train):
        size = (X_train.shape[1] // param) ** 2
        random_pixels = np.random.randint(0, X_train.shape[1] ** 2, size=size)
        X_train = X_train[:, random_pixels]
        return X_train, y_train

    def randset(self, X_train, y_train, param, train):
        if train:
            size = len(X_train) // (param ** 2)
            random_indices = np.random.randint(0, len(X_train), size=size)
            X_train = X_train[random_indices]
            y_train = y_train[random_indices]
        return X_train, y_train