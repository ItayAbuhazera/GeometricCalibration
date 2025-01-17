# geometric_calibrators.py
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from calibrators.base_calibrator import BaseCalibrator
from utils.utils import StabilitySpace, Compression, calc_balanced_acc
from utils.logging_config import setup_logging
from sklearn.neighbors import KDTree
import tensorflow as tf


setup_logging()
logger = logging.getLogger(__name__)


class GeometricCalibrator(BaseCalibrator):
    """
    Class serving as a wrapper for the geometric calibration method (stability/separation).
    """

    def __init__(self, model, X_train, y_train, fitting_func=None, compression_mode=None, compression_param=None,
                 metric='l2', stability_space=None, library='faiss', use_binning=False, n_bins=50):
        """
        Initializes the GeometricCalibrator with a model, stability space, and calibration function.

        Args:
            model: The model to be calibrated (with `predict` and `predict_proba` methods).
            X_train: Training data (flattened images).
            y_train: Training labels.
            fitting_func: Custom fitting function (default: IsotonicRegression).
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling the compression level.
            metric: Distance metric for stability/separation calculations.
            stability_space: Optional custom StabilitySpace instance. If not provided, one is initialized automatically.
            library: The library used for stability calculation (default is 'faiss').
            use_binning (bool): Whether to bin stability scores and calculate average accuracy.
            n_bins (int): Number of bins for stability scores (default: 50).

        """
        super().__init__()
        self.model = model
        self.popt = None
        self._fitted = False
        self.metric = metric.lower()  # Ensure metric is lowercase
        self.use_binning = use_binning
        self.n_bins = n_bins


        # Determine the number of classes (unique labels in y_train)
        self.num_labels = len(np.unique(y_train))  # Fix: Initialize num_labels based on the training labels

        # Default to IsotonicRegression if no custom fitting function is provided
        self.fitting_func = fitting_func if fitting_func else IsotonicRegression(out_of_bounds="clip")

        # Use provided stability space or create a new one with the default settings
        if stability_space:
            self.stab_space = stability_space  # User provided custom StabilitySpace
            logger.info(f"{self.__class__.__name__}: Using custom StabilitySpace provided by user.")
        else:
            # Automatically initialize StabilitySpace with defaults if not provided
            self.stab_space = StabilitySpace(X_train, y_train,
                                             compression=Compression(compression_mode, compression_param),
                                             library=library, metric=self.metric)
            logger.info(f"{self.__class__.__name__}: Initialized StabilitySpace with default settings"
                        f" (library: {library}, metric: {metric}).")

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}"
                    f" and fitting function {self.fitting_func.__class__.__name__}.")

    def fit(self, X_val, y_val):
        """
        Fits the calibrator with the validation data using rounded stability and balanced accuracy.
    
        Args:
            X_val: Validation data (flattened images).
            y_val: Validation labels.
            
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data using balanced accuracy and rounded stability.")
    
        try:
            # Step 1: Predict on validation data
            if hasattr(self.model, "predict_proba"):
                # For sklearn models
                logger.info(f"Sklearn model detected")  # Log a sample of the stability values
                y_pred_val = self.model.predict_proba(X_val)  # Probabilities
                y_pred_classes = self.model.predict(X_val)  # Class labels
            elif hasattr(self.model, "predict"):
                # For tf.keras models
                y_pred_val = self.model.predict(X_val)  # Probabilities
                y_pred_classes = np.argmax(y_pred_val, axis=1)  # Class labels
            else:
                raise ValueError("Model does not support required prediction methods.")

            # Step 2: Compute stability values based on predictions
            stability_val = self.stab_space.calc_stab(X_val, y_pred_val)
            logger.info(f"Stability values (first 200): {stability_val[:200]}")  # Log a sample of the stability values
    
            # Step 3: Round the stability values for binning
            rounded_stability = np.round(stability_val / 10) * 10
            unique_stabilities = np.unique(stability_val)
    
            if self.use_binning:
                # Step 3: Bin stability values
                min_stability, max_stability = np.min(stability_val), np.max(stability_val)
                bin_edges = np.linspace(min_stability, max_stability, self.n_bins + 1)
                bin_indices = np.digitize(stability_val, bins=bin_edges) - 1  # Bin indices start at 0
                
                # Step 4: Compute average accuracy for each bin
                binned_stability = []
                binned_accuracy = []
                for bin_idx in range(self.n_bins):
                    indices_in_bin = np.where(bin_indices == bin_idx)[0]
                    if len(indices_in_bin) > 0:
                        y_true_bin = y_val[indices_in_bin]
                        y_pred_bin = y_pred_classes[indices_in_bin]
                        accuracy = np.mean(y_true_bin == y_pred_bin)
                        binned_stability.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                        binned_accuracy.append(accuracy)

                # Convert to arrays for fitting
                stability_vals = np.array(binned_stability)
                accuracies = np.array(binned_accuracy)
            else:
                # Use raw stability and accuracies without binning
                unique_stabilities = np.unique(stability_val)
                stability_vals = []
                accuracies = []
                for stab in unique_stabilities:
                    indices = np.where(stability_val == stab)[0]
                    y_true_stab = y_val[indices]
                    y_pred_stab = y_pred_classes[indices]
                    acc = np.mean(y_true_stab == y_pred_stab)
                    stability_vals.append(stab)
                    accuracies.append(acc)
                stability_vals = np.array(stability_vals)
                accuracies = np.array(accuracies)
    
            # Step 5: Prepare calibration data: (rounded_stability, accuracy) pairs
            # calibration_data = [(stab, stability_accuracy[stab]) for stab in stability_accuracy]
    
            # Step 6: Fit the provided fitting function (e.g., IsotonicRegression) on the calibration data
            self.popt = self.fitting_func.fit(stability_vals.reshape(-1, 1), accuracies)
            # self.popt = self.fitting_func.fit(np.array(stability_vals).reshape(-1, 1), np.array(accuracies))
    
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted using stability-accuracy pairs and {self.fitting_func.__class__.__name__}.")
    
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
            raise
            
    def calibrate(self, X_test):
        """
        Calibrates the test data based on the fitted model.
    
        Args:
            X_test: Test data (flattened images).
    
        Returns:
            np.ndarray: Calibrated probability matrix for each image and class.
        """
        if not self._fitted:
            raise ValueError("You must fit the calibrator before using it.")
    
        logger.info(f"{self.__class__.__name__}: Calibrating test data.")
    
        try:
            # Predict on the test data using the trained model (get predicted probabilities for all classes)
            # Step 1: Predict on validation data
            if hasattr(self.model, "predict_proba"):
                # For sklearn models
                y_test_pred = self.model.predict_proba(X_test)  # Probabilities
                y_test_labels = self.model.predict(X_test)  # Class labels
            elif hasattr(self.model, "predict"):
                # For tf.keras models
                y_test_pred = self.model.predict(X_test)  # Probabilities
                y_test_labels = np.argmax(y_test_pred, axis=1)  # Class labels
            else:
                raise ValueError("Model does not support required prediction methods.")

            # Initialize progress bar using tqdm
            num_samples = X_test.shape[0]
            num_classes = y_test_pred.shape[1]
            calibrated_probs = np.zeros((num_samples, num_classes))  # Initialize a matrix to store calibrated probabilities
    
            logger.info(f"Starting calibration for {num_samples} samples and {num_classes} classes.")
    
            # Compute stability for the predicted probabilities
            stability_test = self.stab_space.calc_stab(X_test, y_test_pred)
            logger.info(f"Stability values during calibration (first 10): {stability_test[:10]}")  # Add logging
    
            # Apply the fitted calibration function to the stability values
            calibrated_values = self.popt.predict(stability_test.reshape(-1, 1))
            logger.info(f"Calibrated values (first 10): {calibrated_values[:10]}")  # Add logging
    
            # Distribute the calibrated values across the predicted class
            for i in range(X_test.shape[0]):
                # Assign the calibrated probability to the predicted class label
                calibrated_probs[i, y_test_labels[i]] = calibrated_values[i]
                
                # Distribute remaining probability equally across other classes
                remaining_prob = (1 - calibrated_values[i]) / (self.num_labels - 1)
                for j in range(self.num_labels):
                    if j != y_test_labels[i]:
                        calibrated_probs[i, j] = remaining_prob
    
            # Ensure probabilities are in [0, 1] and sum to 1
            calibrated_probs = np.clip(calibrated_probs, 0, 1)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
    
            logger.info(f"{self.__class__.__name__}: Calibration successful.")
    
            return calibrated_probs
    
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Calibration failed with error: {e}")
            raise

class SeparationCalibrator(BaseCalibrator):
    """
    A calibrator based on separation calculations.
    """

    def __init__(self, model, X_train, y_train, compression_mode=None, compression_param=None, metric='l2'):
        """
        Initializes the SeparationCalibrator with a model and separation metrics.

        Args:
            model: The model to be calibrated.
            X_train: Training data.
            y_train: Training labels.
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling compression.
            metric: Distance metric for separation.
        """
        super().__init__()
        self.model = model
        self._fitted = False
        self.separation_space = StabilitySpace(X_train, y_train,
                                                compression=Compression(compression_mode, compression_param),
                                                metric=metric)
        logger.info(f"Initialized {self.__class__.__name__} with separation metrics.")

    def fit(self, X_val, y_val):
        """
        Fits the calibrator using separation metrics.

        Args:
            X_val: Validation data.
            y_val: Validation labels.
        """
        logger.info(f"{self.__class__.__name__}: Fitting using separation metrics.")
        try:
            y_pred_val = self.model.predict(X_val)
            separation_val = self.separation_space.calc_sep(X_val, y_pred_val)

            # Fit isotonic regression based on separation values
            correct = y_val == y_pred_val
            self.popt = IsotonicRegression(out_of_bounds="clip").fit(separation_val, correct)
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted using separation values.")
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit using separation values: {e}")
            raise

    def calibrate(self, X_test):
        """
        Calibrates the test data based on the fitted model.

        Args:
            X_test: Test data.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise ValueError("You must fit the calibrator before using it.")
        logger.info(f"{self.__class__.__name__}: Calibrating based on separation.")

        try:
            y_test_pred = self.model.predict(X_test)
            separation_test = self.separation_space.calc_sep(X_test, y_test_pred)
            calibrated_probs = self.popt.predict(separation_test)
            logger.info(f"{self.__class__.__name__}: Calibration successful.")
            return calibrated_probs
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Calibration failed with error: {e}")
            raise

class GeometricCalibratorTrust(BaseCalibrator):
    def __init__(self, model, X_train, y_train, fitting_func=None, k=10, min_dist=1e-12, 
                 use_binning=False, n_bins=50, use_filtering=False, alpha=0.0):
        """
        Initializes the GeometricCalibratorTrust.
        """
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.min_dist = min_dist
        self.use_binning = use_binning
        self.n_bins = n_bins
        self.use_filtering = use_filtering
        self.alpha = alpha
        self._fitted = False

        # Flatten the training data if necessary
        if len(self.X_train.shape) > 2:
            logger.info(f"Flattening X_train with original shape: {self.X_train.shape}")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
            logger.info(f"Flattened X_train shape: {self.X_train.shape}")

        # Initialize KD-trees for each class
        self.num_labels = len(np.unique(y_train))
        self.kdtrees = [None] * self.num_labels

        # Filter training data if needed
        if self.use_filtering:
            self.filter_by_density()
        else:
            # Initialize KD-trees without filtering
            for label in range(self.num_labels):
                X_label = self.X_train[np.where(self.y_train == label)[0]]
                self.kdtrees[label] = KDTree(X_label)

        # Default to IsotonicRegression if no custom fitting function is provided
        self.fitting_func = fitting_func if fitting_func else IsotonicRegression(out_of_bounds="clip")
        
        logger.info(f"Initialized {self.__class__.__name__} with filtering: {use_filtering}")

    def filter_by_density(self):
        """Filter out points with low kNN density for each class."""
        for label in range(self.num_labels):
            X_label = self.X_train[np.where(self.y_train == label)[0]]
            if len(X_label) > 0:
                kdtree = KDTree(X_label)
                knn_radii = kdtree.query(X_label, k=self.k)[0][:, -1]
                eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
                filtered_indices = np.where(knn_radii <= eps)[0]
                filtered_X = X_label[filtered_indices]
                
                if len(filtered_X) > 0:
                    self.kdtrees[label] = KDTree(filtered_X)
                else:
                    logger.warning(f"No points remained after filtering for class {label}")
                    self.kdtrees[label] = KDTree(X_label)  # Use unfiltered data as fallback

    def calculate_trust_scores(self, X, pred_labels):
        """Calculate trust scores for the given data points and predictions."""
        # Flatten X if necessary
        if len(X.shape) > 2:
            logger.info(f"Flattening X with original shape: {X.shape}")
            X = X.reshape(X.shape[0], -1)
            logger.info(f"Flattened X shape: {X.shape}")

        distances = np.zeros((X.shape[0], self.num_labels))
        for label in range(self.num_labels):
            distances[:, label] = self.kdtrees[label].query(X, k=2)[0][:, -1]
        
        sorted_distances = np.sort(distances, axis=1)
        d_to_pred = distances[range(distances.shape[0]), pred_labels]
        d_to_closest_not_pred = np.where(
            sorted_distances[:, 0] != d_to_pred,
            sorted_distances[:, 0],
            sorted_distances[:, 1]
        )
        
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)

    def fit(self, X_val, y_val):
        """
        Fits the calibrator using validation data.
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data.")
        try:
            # Flatten X_val if the model is not TensorFlow/Keras
            logger.info(f"Original X_val shape: {X_val.shape}")
            if not isinstance(self.model, tf.keras.Model) and len(X_val.shape) > 2:
                X_val = X_val.reshape(X_val.shape[0], -1)
            logger.info(f"Processed X_val shape: {X_val.shape}")

            # Get model predictions
            if hasattr(self.model, "predict_proba"):
                y_pred_val = self.model.predict_proba(X_val)
                y_pred_classes = self.model.predict(X_val)
            elif hasattr(self.model, "predict"):
                y_pred_val = self.model.predict(X_val)
                y_pred_classes = np.argmax(y_pred_val, axis=1)
            else:
                raise ValueError("Model does not support required prediction methods.")

            # Calculate trust scores
            trust_scores = self.calculate_trust_scores(X_val, y_pred_classes)
            logger.info(f"Trust scores (first 10): {trust_scores[:10]}")

            if self.use_binning:
                # Bin trust scores
                min_score, max_score = np.min(trust_scores), np.max(trust_scores)
                bin_edges = np.linspace(min_score, max_score, self.n_bins + 1)
                bin_indices = np.digitize(trust_scores, bins=bin_edges) - 1
                
                # Compute average accuracy for each bin
                binned_scores = []
                binned_accuracy = []
                for bin_idx in range(self.n_bins):
                    indices_in_bin = np.where(bin_indices == bin_idx)[0]
                    if len(indices_in_bin) > 0:
                        y_true_bin = y_val[indices_in_bin]
                        y_pred_bin = y_pred_classes[indices_in_bin]
                        accuracy = np.mean(y_true_bin == y_pred_bin)
                        binned_scores.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                        binned_accuracy.append(accuracy)
                
                scores = np.array(binned_scores)
                accuracies = np.array(binned_accuracy)
            else:
                # Use raw scores
                unique_scores = np.unique(trust_scores)
                scores = []
                accuracies = []
                for score in unique_scores:
                    indices = np.where(trust_scores == score)[0]
                    accuracy = np.mean(y_val[indices] == y_pred_classes[indices])
                    scores.append(score)
                    accuracies.append(accuracy)
                scores = np.array(scores)
                accuracies = np.array(accuracies)

            # Fit the calibration function
            self.popt = self.fitting_func.fit(scores.reshape(-1, 1), accuracies)
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted.")

        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
            raise

    def calibrate(self, X_test):
        """
        Calibrates the test data based on the fitted model.
        
        Args:
            X_test: Test data
            
        Returns:
            Calibrated probability matrix for each sample and class
        """
        if not self._fitted:
            raise ValueError("You must fit the calibrator before using it.")
        
        logger.info(f"{self.__class__.__name__}: Calibrating test data.")
        
        try:
            # Get model predictions
            if hasattr(self.model, "predict_proba"):
                y_test_pred = self.model.predict_proba(X_test)
                y_test_labels = self.model.predict(X_test)
            elif hasattr(self.model, "predict"):
                y_test_pred = self.model.predict(X_test)
                y_test_labels = np.argmax(y_test_pred, axis=1)
            
            # Calculate trust scores
            trust_scores = self.calculate_trust_scores(X_test, y_test_labels)
            
            # Apply calibration function
            calibrated_values = self.popt.predict(trust_scores.reshape(-1, 1))
            
            # Initialize calibrated probabilities matrix
            num_samples = X_test.shape[0]
            calibrated_probs = np.zeros_like(y_test_pred)
            
            # Distribute calibrated values
            for i in tqdm(range(num_samples)):
                pred_class = y_test_labels[i]
                calibrated_probs[i] = y_test_pred[i]  # Copy original probabilities
                calibrated_probs[i] *= calibrated_values[i] / np.sum(calibrated_probs[i])  # Scale by calibrated value
            
            return calibrated_probs

        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to calibrate with error: {e}")
            raise