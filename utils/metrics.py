import numpy as np
from sklearn.metrics import log_loss
import logging
import csv

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """
    A class to calculate various calibration metrics for uncertainty estimation in model predictions.

    Attributes:
        probs (np.ndarray): Calibrated probabilities for each class.
        y_pred (np.ndarray): Predicted class labels.
        y_real (np.ndarray): True class labels.
        n_bins (int): Number of bins to divide probabilities.
    """

    def __init__(self, probs, y_pred, y_real, n_bins=20):
        # Validate inputs
        assert isinstance(probs, np.ndarray), "probs must be a numpy array"
        assert isinstance(y_pred, np.ndarray), "y_pred must be a numpy array"
        assert isinstance(y_real, np.ndarray), "y_real must be a numpy array"
        assert len(probs) == len(y_real) == len(y_pred), "probs, y_pred, and y_real must have the same length"
        assert isinstance(n_bins, int) and n_bins > 0, "n_bins must be a positive integer"

        self.probs = probs
        self.y_pred = y_pred
        self.y_real = y_real
        self.n_bins = n_bins

        # Dictionary of available metrics
        self.metric_functions = {
            "ECE": self.ece,
            "MCE": self.mce,
            "Brier Score": self.brier_score,
            "Log Loss": self.log_loss,
            "ACE": self.ace,
            "Binned Likelihood": self.binned_likelihood,
            "CRSP": self.crsp,
            "Sharpness": self.sharpness,
            "Entropy": self.entropy,
            "NLL": self.nll
        }

    def ece(self):
        """Calculate Expected Calibration Error (ECE)."""
        logger.info("Starting Expected Calibration Error (ECE) calculation.")
        # Handle both 1D and 2D input for probs
        if self.probs.ndim == 2:
            confidence_of_pred_class = np.max(self.probs, axis=1)
        else:
            confidence_of_pred_class = self.probs
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1
        logger.debug(f"Bin indices: {bin_indices[:10]}")
        total_error = 0.0
        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            bin_confidences = confidence_of_pred_class[bin_mask]
            bin_real = self.y_real[bin_mask]
            bin_pred = self.y_pred[bin_mask]

            if len(bin_confidences) > 0:
                bin_acc = np.mean(bin_real == bin_pred)
                bin_conf = np.mean(bin_confidences)
                bin_weight = len(bin_confidences) / len(self.probs)
                total_error += bin_weight * np.abs(bin_acc - bin_conf)

        logger.info(f"Final ECE value: {total_error}")
        return total_error

    def mce(self):
        """Calculate Maximum Calibration Error (MCE)."""
        logger.info("Calculating Maximum Calibration Error.")
        if self.probs.ndim == 2:
            confidence_of_pred_class = np.max(self.probs, axis=1)
        else:
            confidence_of_pred_class = self.probs
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1
        logger.debug(f"Bin indices: {bin_indices[:10]}")
        max_error = 0.0
        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            bin_confidences = confidence_of_pred_class[bin_mask]
            bin_real = self.y_real[bin_mask]
            bin_pred = self.y_pred[bin_mask]

            if len(bin_confidences) > 0:
                bin_acc = np.mean(bin_real == bin_pred)
                bin_conf = np.mean(bin_confidences)
                max_error = max(max_error, np.abs(bin_acc - bin_conf))

        logger.info(f"Final MCE value: {max_error}")
        return max_error

    def brier_score(self):
        """Calculate Brier Score."""
        logger.info("Calculating Brier Score.")
        
        # One-hot encode y_real
        num_classes = self.probs.shape[1]
        y_real_one_hot = np.eye(num_classes)[self.y_real]
        logger.debug(f"Shape of probs: {self.probs.shape}, Shape of y_real_one_hot: {y_real_one_hot.shape}")
        
        # Calculate the Brier Score
        score = np.mean((self.probs - y_real_one_hot) ** 2)
        logger.info(f"Brier Score: {score}")
        return score

    def log_loss(self):
        """Calculate Logarithmic Loss (Log Loss)."""
        logger.info("Calculating Log Loss.")
        return log_loss(self.y_real, self.probs)

    def ace(self):
        """Calculate Average Calibration Error (ACE)."""
        logger.info("Calculating Average Calibration Error (ACE).")
        if self.probs.ndim == 2:
            confidence_of_pred_class = np.max(self.probs, axis=1)
        else:
            confidence_of_pred_class = self.probs
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1

        total_error = 0.0
        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            bin_confidences = confidence_of_pred_class[bin_mask]
            bin_real = self.y_real[bin_mask]
            bin_pred = self.y_pred[bin_mask]

            if len(bin_confidences) > 0:
                bin_acc = np.mean(bin_real == bin_pred)
                bin_conf = np.mean(bin_confidences)
                total_error += np.abs(bin_acc - bin_conf)

        ace_value = total_error / self.n_bins
        logger.info(f"Final ACE value: {ace_value}")
        return ace_value

    def binned_likelihood(self):
        """Calculate Binned Likelihood."""
        logger.info("Calculating Binned Likelihood.")
        if self.probs.ndim == 2:
            confidence_of_pred_class = np.max(self.probs, axis=1)
        else:
            confidence_of_pred_class = self.probs
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1

        total_likelihood = 0.0
        for i in range(self.n_bins):
            bin_mask = bin_indices == i
            bin_confidences = confidence_of_pred_class[bin_mask]
            bin_real = self.y_real[bin_mask]

            if len(bin_confidences) > 0:
                bin_likelihood = np.mean(bin_real * np.log(bin_confidences + 1e-12) +
                                         (1 - bin_real) * np.log(1 - bin_confidences + 1e-12))
                total_likelihood += bin_likelihood

        logger.info(f"Final Binned Likelihood: {total_likelihood}")
        return total_likelihood

    def crsp(self):
        """Calculate Calibration Residual Squared Probability (CRSP)."""
        logger.info("Calculating CRSP.")
        if self.probs.ndim == 2:
            confidence_of_pred_class = np.max(self.probs, axis=1)
        else:
            confidence_of_pred_class = self.probs
        residuals = (confidence_of_pred_class - (self.y_real == self.y_pred).astype(float)) ** 2
        crsp_value = np.mean(residuals)
        logger.info(f"Final CRSP value: {crsp_value}")
        return crsp_value

    def sharpness(self):
        """Calculate Sharpness of the predicted probabilities."""
        logger.info("Calculating Sharpness.")
        sharpness_value = np.mean(np.max(self.probs, axis=1))
        logger.info(f"Final Sharpness value: {sharpness_value}")
        return sharpness_value

    def nll(self):
        """Calculate Negative Log-Likelihood (NLL)."""
        logger.info("Calculating Negative Log-Likelihood (NLL).")
        nll_value = -np.sum(np.log(self.probs[np.arange(len(self.y_real)), self.y_real] + 1e-12)) / len(self.y_real)
        logger.info(f"Final NLL value: {nll_value}")
        return nll_value

    def entropy(self):
        """Calculate Entropy of the predicted probabilities."""
        logger.info("Calculating Entropy.")
        entropy_value = -np.sum(self.probs * np.log(self.probs + 1e-12), axis=1).mean()
        logger.info(f"Final Entropy value: {entropy_value}")
        return entropy_value

    def calculate_all_metrics(self):
        """Calculate all calibration metrics and return as a dictionary."""
        metrics = {}
        for metric_name, metric_func in self.metric_functions.items():
            try:
                metrics[metric_name] = metric_func()
            except Exception as e:
                logger.error(f"Failed to calculate {metric_name}: {e}")
                metrics[metric_name] = None  # Set as None if calculation fails
        return metrics

    def save_metrics_to_csv(self, file_path="metrics.csv"):
        """Calculate all metrics and save results to a CSV file."""
        metrics = self.calculate_all_metrics()
        logger.info(f"Saving metrics to {file_path}")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for metric, value in metrics.items():
                writer.writerow([metric, value])

# Example usage:
# metrics = CalibrationMetrics(probs, y_pred, y_real, n_bins=20)
# print(metrics.calculate_all_metrics())
# metrics.save_metrics_to_csv("calibration_metrics.csv")
