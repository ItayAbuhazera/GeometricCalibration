import numpy as np
from sklearn.metrics import log_loss
import logging

logger = logging.getLogger(__name__)


def ECE_calc(probs, y_pred, y_real, n_bins=20):
    """
    Expected Calibration Error (ECE) calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities for each class.
        y_pred (np.ndarray): Predicted class labels.
        y_real (np.ndarray): True class labels.
        n_bins (int): Number of bins to divide probabilities.

    Returns:
        float: ECE value.
    """
    logger.info("Starting Expected Calibration Error (ECE) calculation.")

    # Select the probabilities of the predicted classes
    confidence_of_pred_class = np.max(probs, axis=1)

    # Bin the confidences
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_of_pred_class, bin_boundaries) - 1

    logger.info(f"Bin boundaries: {bin_boundaries}")
    logger.info(f"Bin indices (first 10): {bin_indices[:10]}")
    
    total_error = 0.0
    total_weight = 0.0  # To track the weight distribution

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_confidences = confidence_of_pred_class[bin_mask]
        bin_real = y_real[bin_mask]
        bin_pred = y_pred[bin_mask]

        if len(bin_confidences) > 0:
            bin_acc = np.mean(bin_real == bin_pred)
            bin_conf = np.mean(bin_confidences)
            bin_weight = len(bin_confidences) / len(probs)
            total_error += bin_weight * np.abs(bin_acc - bin_conf)
            total_weight += bin_weight

            logger.info(f"Bin {i}:")
            logger.info(f"  Bin size: {len(bin_confidences)}")
            logger.info(f"  Accuracy: {bin_acc}")
            logger.info(f"  Confidence: {bin_conf}")
            logger.info(f"  Bin weight: {bin_weight}")
            logger.info(f"  ECE contribution: {bin_weight * np.abs(bin_acc - bin_conf)}")
        else:
            logger.info(f"Bin {i} is empty.")

    logger.info(f"Total weight: {total_weight}")
    logger.info(f"Final ECE value: {total_error}")
    
    return total_error
    

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

def MCE_calc(probs, y_pred, y_real, n_bins=15):
    """
    Maximum Calibration Error (MCE) calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities.
        y_pred (np.ndarray): Predicted class labels.
        y_real (np.ndarray): True class labels.
        n_bins (int): Number of bins to divide probabilities.

    Returns:
        float: MCE value.
    """
    logger.info("Calculating Maximum Calibration Error.")
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_boundaries) - 1

    max_error = 0.0
    for i in range(n_bins):
        bin_probs = probs[bin_indices == i]
        bin_real = y_real[bin_indices == i]
        if len(bin_probs) > 0:
            bin_acc = np.mean(bin_real == y_pred[bin_indices == i])
            bin_conf = np.mean(bin_probs)
            bin_error = np.abs(bin_acc - bin_conf)
            if bin_error > max_error:
                max_error = bin_error
    return max_error


def ACE_calc(probs, y_pred, y_real, n_bins=15):
    """
    Adaptive Calibration Error (ACE) calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities.
        y_pred (np.ndarray): Predicted class labels.
        y_real (np.ndarray): True class labels.
        n_bins (int): Number of bins to divide probabilities.

    Returns:
        float: ACE value.
    """
    logger.info("Calculating Adaptive Calibration Error.")
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_boundaries) - 1

    total_error = 0.0
    for i in range(n_bins):
        bin_probs = probs[bin_indices == i]
        bin_real = y_real[bin_indices == i]
        if len(bin_probs) > 0:
            bin_acc = np.mean(bin_real == y_pred[bin_indices == i])
            bin_conf = np.mean(bin_probs)
            total_error += np.abs(bin_acc - bin_conf)
    return total_error / n_bins


def Brier_score_calc(probs, y_real):
    """
    Brier Score calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities.
        y_real (np.ndarray): True class labels.

    Returns:
        float: Brier Score.
    """
    logger.info("Calculating Brier Score.")
    return np.mean((probs - y_real) ** 2)


def log_loss_calc(probs, y_real):
    """
    Logarithmic Loss (Log Loss) calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities.
        y_real (np.ndarray): True class labels.

    Returns:
        float: Logarithmic Loss value.
    """
    logger.info("Calculating Log Loss.")
    return log_loss(y_real, probs)


def binned_likelihood_calc(probs, y_real, n_bins=15):
    """
    Binned Likelihood (Binned Negative Log Likelihood) calculation.

    Parameters:
        probs (np.ndarray): Calibrated probabilities.
        y_real (np.ndarray): True class labels.
        n_bins (int): Number of bins to divide probabilities.

    Returns:
        float: Binned Negative Log Likelihood.
    """
    logger.info("Calculating Binned Likelihood.")
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_boundaries) - 1

    total_nll = 0.0
    for i in range(n_bins):
        bin_probs = probs[bin_indices == i]
        bin_real = y_real[bin_indices == i]
        if len(bin_probs) > 0:
            likelihoods = bin_real * np.log(bin_probs + 1e-10) + (1 - bin_real) * np.log(1 - bin_probs + 1e-10)
            total_nll -= np.sum(likelihoods)
    return total_nll / len(probs)
