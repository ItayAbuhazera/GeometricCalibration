import logging

import torch
from scipy.optimize import curve_fit
from torch import nn, optim

from calibrators.base_calibrator import BaseCalibrator
import numpy as np

from utils.logging_config import setup_logging
from utils.utils import sigmoid_func

setup_logging()
logger = logging.getLogger(__name__)


class PlattCalibrator(BaseCalibrator):
    def fit(self, y_prob_val, y_true):
        logger.info("Fitting Platt Calibrator.")
        from sklearn.preprocessing import label_binarize

        n_classes = y_prob_val.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        self.sigmoid_params = []

        for i in range(n_classes):
            # Fit sigmoid function for each class
            p0 = [1.0, 0.0]  # Initial guesses for parameters
            probs = y_prob_val[:, i]
            labels = y_true_bin[:, i]
            popt_sigmoid, _ = curve_fit(sigmoid_func, probs, labels, p0, maxfev=1000000)
            self.sigmoid_params.append(popt_sigmoid)

        logger.info("Platt Calibrator fitted successfully for all classes.")

    def calibrate(self, probs):
        logger.info("Calibrating probabilities using Platt Scaling.")

        calibrated_probs = np.zeros_like(probs)

        for i in range(probs.shape[1]):
            popt_sigmoid = self.sigmoid_params[i]
            calibrated_probs[:, i] = sigmoid_func(probs[:, i], *popt_sigmoid)

        # Normalize calibrated probabilities to sum to 1
        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)

        logger.debug(f"Calibrated probabilities (sample): {calibrated_probs[:10]}")

        return calibrated_probs



class TemperatureScalingCalibrator(BaseCalibrator):
    """ Maximum likelihood temperature scaling (Guo et al., 2017) """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_trace = None
        logger.info(f"Initialized TSCalibrator with temperature={temperature}")

    def fit(self, logits, y):
        """ Fits temperature scaling using hard labels. """
        logger.info("TemperatureScalingCalibrator: Starting the fitting process.")
        try:
            self.n_classes = logits.shape[1]
            _model_logits = torch.from_numpy(logits)
            _y = torch.from_numpy(y)
            _temperature = torch.tensor(self.temperature, requires_grad=True)

            nll = nn.CrossEntropyLoss()
            optimizer = optim.Adam([_temperature], lr=0.05)
            loss_trace = []

            logger.info("TemperatureScalingCalibrator: Beginning optimization.")
            for step in range(7500):
                optimizer.zero_grad()
                loss = nll(_model_logits / _temperature, _y)
                loss.backward()
                optimizer.step()
                loss_trace.append(loss.item())
                with torch.no_grad():
                    _temperature.clamp_(min=1e-2, max=1e4)

                if np.abs(_temperature.grad) < 1e-3:
                    logger.info("TemperatureScalingCalibrator: Convergence reached.")
                    break

            self.loss_trace = loss_trace
            self.temperature = _temperature.item()
            logger.info(f"TemperatureScalingCalibrator: Fitting complete. Final temperature: {self.temperature}")
        except Exception as e:
            logger.error(f"TemperatureScalingCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        """ Apply temperature scaling to probabilities. """
        logger.info("TemperatureScalingCalibrator: Calibrating probabilities.")
        try:
            calibrated_probs = probs ** (1. / self.temperature)
            calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)
            return calibrated_probs
        except Exception as e:
            logger.error(f"TemperatureScalingCalibrator: Calibration failed with error: {e}")
            raise
