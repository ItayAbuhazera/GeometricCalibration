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
from sklearn.neighbors import KDTree, NearestNeighbors  # Add this line

import faiss
import numpy as np
from scipy.optimize import optimize, minimize
import h5py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
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

    def compute_stab(self, whom, y_pred, metric='l2'):
        """
        Compute stability for the given data split ('test' or 'val') using a specific metric.

        Parameters:
            whom (str): Dataset split to compute stability for ('test' or 'val').
            y_pred (np.ndarray): Predicted labels.
            metric (str): Distance metric to use ('l2' by default).

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
    Add small random noise to a matrix or scalar and normalize it.

    Parameters:
        matrix (np.ndarray or float): Matrix or scalar to nudge.
        delta (float): Amount of noise to add.

    Returns:
        np.ndarray or float: Nudged matrix or scalar.
    """
    logger.debug(f"Nudging input with delta {delta}.")
    
    if np.isscalar(matrix):  # Handle scalar input
        noise = np.random.uniform(low=0, high=delta)
        return (matrix + noise) / (1 + delta)
    else:  # Handle array input
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


def stability_calc_knn(trainX, testX, train_y, test_y_pred, num_labels, metric='l2'):
    """
    Calculate stability using k-Nearest Neighbors (kNN) for a given dataset.

    Parameters:
        trainX (np.ndarray): Training data.
        testX (np.ndarray): Test data.
        train_y (np.ndarray): Training labels.
        test_y_pred (np.ndarray): Predicted labels for the test set.
        num_labels (int): Number of labels/classes.
        metric (str): Distance metric to use for nearest neighbors (default: 'l2').

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

    def __init__(self, X_train, y_train, compression=None, library='knn', metric='l2', num_labels=None,
                 faiss_mode='exact', nlist=None):
        """
        Initialize the stability space with consistent compression handling.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing StabilitySpace with {library} library and {metric} metric.")

        self.metric = metric.lower()
        self.library = library
        self.num_labels = num_labels or len(set(y_train))
        self.compression = compression
        self.faiss_mode = faiss_mode
        self.nlist = nlist or self.num_labels

        # Store original shapes for logging
        original_shape = X_train.shape

        # Apply compression if provided
        if self.compression:
            self.logger.info("Applying compression to training data.")
            X_train, y_train = self.compression(X_train, y_train)
            self.logger.info(
                f"Compression applied. Original shape: {original_shape}, Compressed shape: {X_train.shape}")
            # Store compression input/output shapes for validation
            self.input_shape = original_shape[1:]
            self.output_shape = X_train.shape[1:]
        else:
            self.input_shape = original_shape[1:]
            self.output_shape = original_shape[1:]

        # Ensure data is properly shaped for the chosen library
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            self.logger.info(f"Reshaped training data to 2D: {X_train.shape}")

        self.X_train = X_train
        self.y_train = y_train

        # Initialize appropriate model based on library choice
        if library == 'faiss':
            self._initialize_faiss()
        elif library == 'knn':
            self._initialize_knn()
        elif library == 'kdtree':
            self._initialize_kdtree()
        elif library == 'separation':
            self.logger.info("Using separation-based stability calculation.")
        else:
            raise ValueError(f"Unsupported library: {library}")

    def _get_distance(self, x, y, metric):
        """
        Compute distance between two points using specified metric.
        
        Args:
            x, y: Points to compute distance between
            metric: Distance metric to use
        Returns:
            float: Distance between points
        """
        if metric == 'cosine':
            dot_product = np.dot(x, y)
            norms = np.linalg.norm(x) * np.linalg.norm(y)
            return 1 - (dot_product / norms if norms != 0 else 0)
        else:
            norm_val = self._get_norm_value(metric)
            return np.linalg.norm(x - y, ord=norm_val)

    def _get_norm_value(self, metric):
        """Get the numpy norm value for the given metric."""
        norm_map = {
            'l1': 1,
            'l2': 2,
            'linf': np.inf
        }
        return norm_map.get(metric.lower(), 2)  # Default to L2 if metric not found
    
    def _get_faiss_index(self, dim, metric='l2'):
        """
        Get appropriate FAISS index based on metric.
        
        Args:
            dim: Dimension of the vectors
            metric: Distance metric to use
        Returns:
            FAISS index object
        """
        if metric == 'l2':
            return faiss.IndexFlatL2(dim)
        elif metric == 'l1':
            return faiss.IndexFlat(dim, faiss.METRIC_L1)
        elif metric == 'linf':
            return faiss.IndexFlat(dim, faiss.METRIC_Linf)
        elif metric == 'cosine':
            return faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        else:
            raise ValueError(f"Unsupported FAISS metric: {metric}")


    def _initialize_kdtree(self):
        """
        Initialize KDTrees for each label, similar to TrustScore implementation.
        """
        self.logger.info("Initializing KDTree indices for each class.")
        
        # Flatten X_train if it has more than 2 dimensions
        if len(self.X_train.shape) > 2:
            self.logger.debug("Flattening X_train for KDTree compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)

        # Initialize KDTrees for each class
        self.kdtrees = [None] * self.num_labels
        for label in range(self.num_labels):
            # Get points for current class
            idx_same = np.where(self.y_train == label)[0]
            if len(idx_same) > 0:
                X_label = self.X_train[idx_same]
                self.kdtrees[label] = KDTree(X_label, metric=self.metric)
                self.logger.debug(f"KDTree initialized for class {label} with {len(idx_same)} points")
            else:
                self.logger.warning(f"No points found for class {label}")


    
    def _initialize_faiss(self):
        """
        Initialize FAISS indices for each label, supporting both exact and approximate modes.
        """
        self.same_nbrs = {}
        self.other_nbrs = {}

        # Ensure X_train is 2D for FAISS
        if len(self.X_train.shape) > 2:
            self.logger.debug("Flattening X_train for FAISS compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).astype('float32')

        # Normalize vectors if using cosine similarity
        if self.metric == 'cosine':
            self.logger.debug("Normalizing vectors for cosine similarity.")
            faiss.normalize_L2(self.X_train)

        dim = self.X_train.shape[1]
        for label in range(self.num_labels):
            idx_same = np.where(self.y_train == label)[0]
            idx_other = np.where(self.y_train != label)[0]

            if self.faiss_mode == 'exact':
                # Exact FAISS indices
                self.same_nbrs[label] = self._get_faiss_index(dim, self.metric)
                self.same_nbrs[label].add(self.X_train[idx_same])

                self.other_nbrs[label] = self._get_faiss_index(dim, self.metric)
                self.other_nbrs[label].add(self.X_train[idx_other])

            elif self.faiss_mode == 'approximate':
                # Approximate FAISS indices with clustering
                base_index = self._get_faiss_index(dim, self.metric)

                # Ensure nlist is smaller than or equal to the number of training points
                adjusted_nlist_same = min(self.nlist, max(1, len(idx_same) // 39))
                adjusted_nlist_other = min(self.nlist, max(1, len(idx_other) // 39))

                # For same_nbrs
                if len(idx_same) > 0:
                    self.same_nbrs[label] = faiss.IndexIVFFlat(base_index, dim, adjusted_nlist_same)
                    self.same_nbrs[label].train(self.X_train[idx_same])
                    self.same_nbrs[label].add(self.X_train[idx_same])
                    self.same_nbrs[label].nprobe = min(adjusted_nlist_same // 4, adjusted_nlist_same)
                else:
                    self.logger.warning(f"No points for label {label} in same_nbrs.")
                # For other_nbrs
                if len(idx_other) > 0:
                    self.other_nbrs[label] = faiss.IndexIVFFlat(base_index, dim, adjusted_nlist_other)
                    self.other_nbrs[label].train(self.X_train[idx_other])
                    self.other_nbrs[label].add(self.X_train[idx_other])
                    self.other_nbrs[label].nprobe = min(adjusted_nlist_other // 4, adjusted_nlist_other)
                else:
                    self.logger.warning(f"No points for label {label} in other_nbrs.")
            else:
                raise ValueError(f"Unsupported FAISS mode: {self.faiss_mode}")
            

    def _initialize_knn(self):
        """
        Initialize NearestNeighbors models for each label, flattening input if necessary.
        """
        self.same_nbrs = []
        self.other_nbrs = []
        if self.metric == "linf":
            self.metric = "chebyshev"
    
        # Flatten X_train if it has more than 2 dimensions (e.g., images)
        if len(self.X_train.shape) > 2:
            self.logger.debug("Flattening X_train for KNN compatibility.")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1).astype('float32')
    
        for label in range(self.num_labels):
            idx_same = np.where(self.y_train == label)[0]
            idx_other = np.where(self.y_train != label)[0]
    
            # Initialize NearestNeighbors for KNN library
            same_nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.X_train[idx_same])
            other_nn = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(self.X_train[idx_other])
    
            self.same_nbrs.append(same_nn)
            self.other_nbrs.append(other_nn)

    def _stability_kdtree(self, valX, val_y_pred):
        """
        Calculate stability using KDTrees with efficient distance calculations.
        Similar to TrustScore's approach but adapted for stability calculation.
        """
        self.logger.info("Calculating stability using KDTree.")
        
        # Ensure input is 2D
        valX = valX.reshape(valX.shape[0], -1)
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred
        
        # Initialize distance matrix
        distances = np.zeros((len(valX), self.num_labels))
        
        # Calculate distances to each class
        for label in range(self.num_labels):
            if self.kdtrees[label] is not None:
                # Get distance to 2nd nearest neighbor (k=2) as in TrustScore
                distances[:, label] = self.kdtrees[label].query(valX, k=2)[0][:, -1]
        
        # Calculate stability scores
        stability = np.zeros(len(valX))
        for i in tqdm(range(len(valX)), desc="Calculating Stability (KDTree)", unit="sample"):
            pred_label = predicted_labels[i]
            
            # Get distance to predicted class
            d_to_pred = distances[i, pred_label]
            
            # Get distances to other classes
            other_distances = distances[i, :]
            other_distances[pred_label] = np.inf  # Exclude predicted class
            d_to_closest_not_pred = np.min(other_distances)
            
            # Calculate stability as in original implementation
            stability[i] = (d_to_closest_not_pred - d_to_pred) / 2
        
        return stability

    def _stability_knn(self, valX, val_y_pred):
        """
        Calculate stability using KNN with a progress indicator.
        Ensures validation data matches the training data dimensions.
        """
        self.logger.info("Calculating stability using KNN.")
        stability = np.zeros(len(valX))
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred

        # Get the expected feature dimension from the training data
        expected_features = self.X_train.shape[1]

        # Reshape validation data to match training dimensions
        if valX.shape[1] != expected_features:
            self.logger.warning(
                f"Validation data shape {valX.shape[1]} does not match training shape {expected_features}. "
                "This might indicate a compression mismatch.")
            return np.zeros(len(valX))  # Return zeros instead of raising an error

        for i in tqdm(range(len(valX)), desc="Calculating Stability (KNN)", unit="sample"):
            x = valX[i].reshape(1, -1)
            pred_label = int(predicted_labels[i])

            try:
                dist_same, _ = self.same_nbrs[pred_label].kneighbors(x)
                dist_other, _ = self.other_nbrs[pred_label].kneighbors(x)
                stability[i] = (dist_other[0][0] - dist_same[0][0]) / 2
            except Exception as e:
                self.logger.error(f"Error in KNN stability calculation for sample {i}: {e}")
                stability[i] = np.nan

        return stability

    def _stability_faiss(self, valX, val_y_pred):
        """
        Calculate stability using FAISS with a progress indicator.
        Ensures validation data matches the training data dimensions.
        """
        self.logger.info("Calculating stability using FAISS.")
        stability = np.zeros(len(valX))
        predicted_labels = np.argmax(val_y_pred, axis=1) if len(val_y_pred.shape) > 1 else val_y_pred

        # Get the expected feature dimension from the training data
        expected_features = self.X_train.shape[1]

        # Check if validation data has the correct shape
        if valX.shape[1] != expected_features:
            self.logger.warning(
                f"Validation data shape {valX.shape[1]} does not match training shape {expected_features}. "
                "This might indicate a compression mismatch.")
            return np.zeros(len(valX))  # Return zeros instead of raising an error

        # Prepare validation data
        valX = valX.reshape(len(valX), -1).astype('float32')
        if self.metric == 'cosine':
            faiss.normalize_L2(valX)

        for i in tqdm(range(len(valX)), desc="Calculating Stability (FAISS)", unit="sample"):
            x = valX[i:i + 1]  # Keep as 2D array
            pred_label = int(predicted_labels[i])

            try:
                _, dist_same = self.same_nbrs[pred_label].search(x, 1)
                _, dist_other = self.other_nbrs[pred_label].search(x, 1)

                # For cosine similarity, convert similarity to distance
                if self.metric == 'cosine':
                    dist_same = 1 - dist_same
                    dist_other = 1 - dist_other

                stability[i] = (dist_other[0][0] - dist_same[0][0]) / 2
            except Exception as e:
                self.logger.error(f"Error in FAISS stability calculation for sample {i}: {e}")
                stability[i] = np.nan

        return stability

    def _stability_separation(self, testX, pred_y, norm='L2', parallel=False):
        """
        Calculate separation-based stability with progress tracking.
        """
        self.logger.info("Calculating stability using separation method.")
        
        if parallel:
            self.logger.debug("Entering parallel separation calculation with progress tracking.")
            
            # Use a progress bar to track the entire operation in parallel mode
            results = list(tqdm(self._sep_calc_parallel(testX, pred_y, norm=norm), 
                                desc="Calculating Separation (Parallel)", unit="sample"))
            
            self.logger.debug("Completed parallel separation calculation.")
            return np.array(results)
        else:
            self.logger.debug("Entering sequential separation calculation with progress tracking.")
            
            # Call _sep_calc with tqdm integrated for sequential calculations
            results = np.array(self._sep_calc(testX, pred_y, norm=norm))
            
            self.logger.debug("Completed sequential separation calculation.")
            return results

    def _sep_calc_parallel(self, testX, pred_y, norm='L2'):
        """
        Calculate the separation of all test/val examples in parallel with progress tracking.
        """
        print("Entered _sep_calc_parallel")
        self.logger.debug("Starting parallel separation calculation.")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each task to the executor and track progress with tqdm
            futures = [
                executor.submit(self._sep_calc_point, x, self.X_train, self.y_train, pred, norm)
                for x, pred in zip(testX, pred_y)
            ]
            
            print(f"Submitted {len(futures)} tasks to the executor.")
            self.logger.debug(f"Submitted {len(futures)} tasks to the executor.")
            
            # Use tqdm to show progress as futures complete
            separation = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                               desc="Calculating Separation (Parallel)", unit="sample"):
                try:
                    result = future.result()  # Get the result of each future
                    separation.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel separation calculation: {e}")
            
            print("Completed gathering results from futures.")
            self.logger.debug("Completed gathering results from futures.")
            
        return separation

    def _sep_calc(self, testX, pred_y, norm):
        """
        Calculate the separation of all test/val examples without parallel processing, with progress tracking.
        """
        print("Started _sep_calc with tqdm progress bar")
        self.logger.debug("Starting sequential separation calculation with tqdm progress bar.")
        
        # Use tqdm to track progress over testX for the sequential calculation
        results = []
        for i, x in tqdm(enumerate(testX), desc="Calculating Separation (Sequential)", unit="sample", total=len(testX)):
            result = self._sep_calc_point(x, self.X_train, self.y_train, pred_y[i], norm)
            results.append(result)
            
        print("Completed _sep_calc with tqdm progress bar")
        self.logger.debug("Completed sequential separation calculation with tqdm progress bar.")
        return results

    def _sep_calc_point(self, x, X_train, y_train, y_pred, norm='L2'):
        """
        Calculate the separation for a single test instance.
        """
        self.logger.debug("Started _sep_calc_point")
        
        # Ensure `y_pred` is a scalar
        if hasattr(y_pred, "__len__") and len(y_pred) > 1:
            y_pred = np.argmax(y_pred)  # Convert probability vector to a class label if needed
    
        # Flatten `x` if it has more than one dimension
        if x.ndim > 1:
            x = x.flatten()
        
        same = [(self._get_distance(x, train.flatten(), self.metric), index) 
                for index, train in enumerate(X_train) if y_train[index] == y_pred]
        others = [(self._get_distance(x, train.flatten(), self.metric), index) 
                  for index, train in enumerate(X_train) if y_train[index] != y_pred]
        
        same.sort(key=lambda x: x[0])
        others.sort(key=lambda x: x[0])
        
        min_r = same[0][0] + 2 * others[0][0]
        sep_other = min_r
        for o in others:
            sep_same = np.NINF
            if o[0] > min_r:
                break
            for s in same:
                if s[0] > min(min_r, o[0]) and o[0] > same[0][0]:
                    break
                x_s = X_train[s[1]].flatten()  # Ensure x_s is also flattened
                x_o = X_train[o[1]].flatten()  # Ensure x_o is also flattened
                sep_same = max(self._two_point_sep_calc(x, x_s, x_o), sep_same)
            sep_other = min(sep_same, sep_other)
            min_r = same[0][0] + 2 * max(0, sep_other)
        
        self.logger.debug("Completed _sep_calc_point")
        
        return sep_other

    def _two_point_sep_calc(self, x, x1, x2):
        """
        Calculate the separation parameter for a single test point and two nearest points.
        """
        a = self._get_distance(x, x1, self.metric)
        b = self._get_distance(x, x2, self.metric)
        c = self._get_distance(x1, x2, self.metric)
        return ((b ** 2 - a ** 2) / (2 * c))

    def calc_stab(self, X_val, y_val_pred, timeout=1800):
        """
        Calculate stability for the validation set with proper compression handling.
        """
        start_time = time.time()

        # Apply compression if provided
        if self.compression:
            logger.info("Applying compression before stability calculation")
            X_val_original_shape = X_val.shape
            X_val_compressed, _ = self.compression(X_val, None, train=False)
            logger.info(f"Compressed validation data from {X_val_original_shape} to {X_val_compressed.shape}")
            X_val = X_val_compressed  # Use compressed data for stability calculation

        # Ensure data is properly shaped
        if len(X_val.shape) > 2:
            X_val = X_val.reshape(X_val.shape[0], -1)
            self.logger.info(f"Reshaped validation data to 2D: {X_val.shape}")

        # Verify shapes match
        if X_val.shape[1] != self.X_train.shape[1]:
            self.logger.error(f"Validation data shape {X_val.shape[1]} does not match "
                              f"training shape {self.X_train.shape[1]}")
            return np.zeros(len(X_val))

        # Calculate stability using appropriate method
        if self.library == 'faiss':
            stability = self._stability_faiss(X_val, y_val_pred)
        elif self.library == 'knn':
            stability = self._stability_knn(X_val, y_val_pred)
        elif self.library == 'kdtree':
            stability = self._stability_kdtree(X_val, y_val_pred)
        elif self.library == 'separation':
            stability = self._stability_separation(X_val, y_val_pred)
        else:
            raise ValueError(f"Unsupported library: {self.library}")

        elapsed_time = time.time() - start_time
        self.logger.info(f"Time taken for stability calculation: {elapsed_time:.2f} seconds")

        return stability


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
    """

    def __init__(self, compression_types, compression_params):
        """
        Initialize compression with single type and parameter or lists of them.

        Parameters:
            compression_types: Single compression type or list of types
            compression_params: Single parameter or list of parameters
        """
        # Convert single values to lists for consistent handling
        self.compression_types = [compression_types] if isinstance(compression_types, str) else compression_types
        self.compression_params = [compression_params] if isinstance(compression_params, (int, float)) else compression_params
        self.pca_model = None

        logging.info(f"Initialize Compression with {self.compression_types} and {self.compression_params}")

    def __call__(self, X_train, y_train, train=True):
        """
        Apply compression to the input data.

        Parameters:
            X_train: Input data, can be either:
                     - 2D (n_samples, n_features), or
                     - 4D (n_samples, height, width, channels).
            y_train: Corresponding labels.
            train: Whether this call is for the training set (affects PCA fitting, etc.)

        Returns:
            (X_compressed, y_compressed): The compressed data and (optionally) filtered labels.
        """
        logger.info(f"Input data shape: {X_train.shape}, dtype: {X_train.dtype}")
        logger.info(f"Input dimensionality check:")
        logger.info(f"- Number of dimensions: {len(X_train.shape)}")

        # Step A: Convert (N, 3072) => (N, 32, 32, 3) if we suspect it is CIFAR-like.
        # Or, if it's (N, 784) => (N, 28, 28, 1), etc.
        # You can check for multiple known shapes if you like. For example:
        #  3072 = 32*32*3  => color 32x32
        #  1024 = 32*32    => grayscale 32x32
        #  784  = 28*28    => grayscale 28x28
        #  ...
        X_reshaped = X_train
        if len(X_train.shape) == 2:
            # Attempt to un-flatten if it matches known shapes.
            num_features = X_train.shape[1]

            # For CIFAR (RGB, 32x32):
            if num_features == 32*32*3:
                logger.info("Detected flattened 32x32x3 input; reshaping to (N, 32, 32, 3).")
                X_reshaped = X_train.reshape(-1, 32, 32, 3)
            # For MNIST-like single-channel 28x28:
            elif num_features == 28*28:
                logger.info("Detected flattened 28x28 input; reshaping to (N, 28, 28, 1).")
                X_reshaped = X_train.reshape(-1, 28, 28, 1)
            # For grayscale 32x32:
            elif num_features == 32*32:
                logger.info("Detected flattened 32x32 input; reshaping to (N, 32, 32, 1).")
                X_reshaped = X_train.reshape(-1, 32, 32, 1)
            else:
                logger.info("2D input does not match a known shape; leaving as (N, features).")

        # Next, decide whether we truly keep 4D or flatten single-channel
        if len(X_reshaped.shape) == 4:
            batch_size, height, width, channels = X_reshaped.shape
            logger.info(f"Detected 4D input: batch={batch_size}, height={height}, width={width}, channels={channels}")

            # If single channel, flatten to (N, height*width).
            # If multiple channels, keep the 4D shape.
            if channels == 1:
                X_reshaped = X_reshaped.reshape(batch_size, height * width)
                logger.info(f"Reshaped single-channel 4D to 2D: {X_reshaped.shape}")
            else:
                logger.info(f"Keeping multi-channel images in 4D. Shape remains: {X_reshaped.shape}")

        elif len(X_reshaped.shape) == 2:
            logger.info("Detected 2D input (no further reshape).")

        else:
            # If there's an unexpected number of dimensions, just return as-is
            logger.warning("Input data is neither 2D nor 4D. Returning unmodified.")
            return X_train, y_train

        # Now we have X_reshaped which is either:
        #  - (N, H*W) for single-channel
        #  - (N, H, W, C) for multi-channel
        #  - (N, features) if unrecognized shape

        X_compressed = X_reshaped

        # Attempt to detect 'pixels' for certain compressions if we have 2D single-channel data
        pixels = None
        if len(X_compressed.shape) == 2:
            num_features = X_compressed.shape[1]
            # If it's a perfect square, set pixels
            if sqrt(num_features).is_integer():
                pixels = int(sqrt(num_features))
                logger.info(f"2D input is a perfect square of size {pixels}x{pixels}.")
            else:
                # e.g. 3072 is not a perfect square, but let's do nothing here
                pass

        # ---------------------------------------------------------
        #  Now apply the requested compression(s)
        # ---------------------------------------------------------
        for comp_type, param in zip(self.compression_types, self.compression_params):
            logger.info(f"\nApplying {comp_type} compression with parameter {param}")
            logger.info(f"Before compression shape: {X_compressed.shape}")

            if comp_type == 'Avgpool':
                pooling = torch.nn.AvgPool2d(param)
                # If 4D => apply (N, H, W, C) => (N, C, H, W)
                if len(X_compressed.shape) == 4:
                    batch_size, h, w, c = X_compressed.shape
                    X_for_pool = torch.tensor(X_compressed.transpose(0, 3, 1, 2), dtype=torch.float32)
                    X_pooled = pooling(X_for_pool)  # => (N, C, outH, outW)
                    X_pooled_np = X_pooled.numpy().transpose(0, 2, 3, 1)
                    X_compressed = X_pooled_np.reshape(batch_size, -1)

                elif len(X_compressed.shape) == 2 and pixels is not None:
                    # (N, pixels*pixels) => reshape => pool => flatten
                    X_for_pool = torch.tensor(X_compressed.reshape(-1, 1, pixels, pixels), dtype=torch.float32)
                    X_pooled = pooling(X_for_pool)
                    X_compressed = X_pooled.reshape(len(X_compressed), -1).numpy()
                else:
                    logger.warning("Avgpool: Unable to determine structure for pooling; skipping.")

                logger.info(f"Applied {comp_type}. New shape: {X_compressed.shape}")

            elif comp_type == 'Maxpool':
                try:
                    pooling = torch.nn.MaxPool2d(param)
                    if len(X_compressed.shape) == 4:
                        # (N, H, W, C) => (N, C, H, W)
                        batch_size, h, w, c = X_compressed.shape
                        logger.info(f"Detected 4D input for Maxpool: (batch={batch_size}, height={h}, width={w}, channels={c})")

                        X_for_pool = torch.tensor(X_compressed.transpose(0, 3, 1, 2), dtype=torch.float32)
                        X_pooled = pooling(X_for_pool)  # => (N, C, pooledH, pooledW)

                        # Convert back
                        pooled_height, pooled_width = X_pooled.shape[-2], X_pooled.shape[-1]
                        X_compressed = X_pooled.numpy().transpose(0, 2, 3, 1)
                        logger.info(f"After pooling (4D) => {X_compressed.shape}")

                        # Flatten
                        X_compressed = X_compressed.reshape(batch_size, -1)
                        logger.info(f"Final flattened shape after Maxpool: {X_compressed.shape}")

                    elif len(X_compressed.shape) == 2 and pixels is not None:
                        logger.info(f"Detected 2D input: {X_compressed.shape}, using {pixels}x{pixels} spatial dims")
                        X_for_pool = X_compressed.reshape(-1, 1, pixels, pixels)
                        X_pooled = pooling(torch.tensor(X_for_pool, dtype=torch.float32))
                        X_compressed = X_pooled.reshape(len(X_compressed), -1).numpy()
                        logger.info(f"Final shape after Maxpool (2D => 2D): {X_compressed.shape}")
                    else:
                        logger.warning("Maxpool: Unable to determine 2D/4D structure properly; skipping.")

                except Exception as e:
                    logger.error(f"Error during Maxpool: {str(e)}")
                    logger.error(f"Current shapes - X_compressed: {X_compressed.shape}")
                    return X_train, y_train

            elif comp_type == 'resize':
                # Use TensorFlow's image.resize
                if pixels is None:
                    logger.warning("Cannot apply 'resize' if input is not a known square or 4D color.")
                    continue
                size = pixels // param
                X_compressed = tf.image.resize(
                    X_compressed.reshape(len(X_compressed), pixels, pixels)[..., np.newaxis],
                    [size, size]
                ).numpy().reshape(len(X_compressed), -1)
                pixels = size
                logging.info(f"Applied {comp_type} compression. New shape: {X_compressed.shape}")

            elif comp_type == 'PCA':
                # Basic PCA compression
                if len(X_compressed.shape) != 2:
                    logger.warning("PCA requires a 2D input of shape (n_samples, n_features). Skipping.")
                    continue
                # param is how we reduce dimensionality
                # e.g., if param=2, we might do half the dimensions, etc.
                # Customize how you set n_components:
                size = pixels // param if pixels else 8  # example fallback
                n_components = size * size

                if train:
                    self.pca_model = PCA(n_components=n_components)
                    X_compressed = self.pca_model.fit_transform(X_compressed)
                else:
                    if self.pca_model is None:
                        raise ValueError("PCA model must be fitted before transform.")
                    X_compressed = self.pca_model.transform(X_compressed)

                logging.info(f"Applied {comp_type} compression. New shape: {X_compressed.shape}")

            elif comp_type == 'randpix':
                # Randomly select (pixels // param)^2 pixel indices
                if pixels is None:
                    logger.warning("randpix requires a 2D input of shape (n_samples, pixels^2). Skipping.")
                    continue
                size = (pixels // param) ** 2
                random_pixels = np.random.randint(0, pixels * pixels, size=size)
                X_compressed = X_compressed[:, random_pixels]
                pixels = int(sqrt(size))
                logging.info(f"Applied {comp_type} compression. New shape: {X_compressed.shape}")

            elif comp_type == 'randset':
                # Randomly select a subset of data
                if train:
                    size = len(X_compressed) // (param ** 2)
                    random_indices = np.random.randint(0, len(X_compressed), size=size)
                    X_compressed = X_compressed[random_indices]
                    y_train = y_train[random_indices]
                    logging.info(f"Applied {comp_type} compression. New shape: {X_compressed.shape}")

            else:
                logging.info(f'No valid compression method found for {comp_type}. Skipping.')
                return X_compressed, y_train

            logger.info(f"After {comp_type} compression:")
            logger.info(f"- Output shape: {X_compressed.shape}")
            logger.info(f"- Output dtype: {X_compressed.dtype}")

        return X_compressed, y_train

