# Calibrato

**Calibrato** is an open-source Python library designed for **model calibration**, with a focus on **geometric calibration methods**. The library also includes additional calibration techniques, such as isotonic regression, Platt scaling, temperature scaling, and trust score calibration, making it a comprehensive tool for improving the reliability of probabilistic predictions from machine learning models.

## Key Features

- **Geometric Calibration**: Novel calibration methods leveraging geometric properties with libraries like FAISS and KNN.
- **Transformations**: Support for rotating, shifting, and adding noise to test sets for robustness testing.
- **Parametric Calibrators**: Includes Platt scaling, isotonic regression, and temperature scaling.
- **Trust Score Calibration**: Provides filtered and unfiltered trust score calibration.
- **Support for Various Models**: Compatible with both neural networks (e.g., CNN, DenseNet) and traditional ML models (e.g., Random Forest, Gradient Boosting).
- **Multi-Dataset Compatibility**: Preconfigured loaders for popular datasets like MNIST, CIFAR-10, CIFAR-100, and Fashion MNIST.
- **Comprehensive Metrics**: Includes tools for Expected Calibration Error (ECE), accuracy, and more.
- **Customizable Framework**: Easily add new calibration techniques or models.

## Installation

To install the library, clone the repository and install the required dependencies:
'''bash
git clone https://github.com/ItayAbuhazera/Calibrato.git
cd Calibrato
pip install -r requirements.txt
'''
## Usage

Here’s how to use **Calibrato** for geometric calibration and other calibration methods:

### 1. Initialize and Prepare Data
Load a dataset, split it into train, validation, and test sets, and apply optional transformations.

from utils.utils import prepare_data

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
    dataset_name="cifar10", 
    random_state=42, 
    transformed=True
)

### 2. Train or Load a Model
Train or load a neural network (e.g., CNN or DenseNet) or other machine learning models.

from utils.utils import prepare_model

model = prepare_model(
    X_train, y_train, X_val, y_val, 
    dataset_name="cifar10", 
    random_state=42, 
    model_type="cnn", 
    dataset_epochs={"cifar10": 35}
)

### 3. Apply Geometric Calibration
Calibrate the model using geometric methods like FAISS, KNN, or separation.
'''python

from calibrators.geometric_calibrators import GeometricCalibrator

geo_calibrator = GeometricCalibrator(
    model=model, 
    X_train=X_train, 
    y_train=y_train, 
    library="faiss", 
    metric="L2"
)
geo_calibrator.fit(X_val, y_val)
'''
# Calibrate test predictions
calibrated_probs = geo_calibrator.calibrate(X_test)

### 4. Calculate Metrics
Evaluate the calibrated predictions using metrics like ECE or accuracy.
'''python
from utils.metrics import CalibrationMetrics

metrics = CalibrationMetrics(calibrated_probs, y_test_pred, y_test, n_bins=20)
ece = metrics.calculate_ece()
print(f"Expected Calibration Error (ECE): {ece}")
'''
### 5. Run End-to-End Pipeline
You can run the entire calibration pipeline from the command line:

python main.py --dataset_name cifar10 --random_state 42 --model_type cnn --metric L2 --transformed

## Supported Calibration Methods

1. **Geometric Methods**:
   - FAISS (Exact and Approximate)
   - KNN
   - Separation-based Calibration

2. **Parametric Methods**:
   - Platt Scaling
   - Isotonic Regression
   - Temperature Scaling

3. **Trust Score Calibration**:
   - Filtered
   - Unfiltered

## Contributing

We welcome contributions! To contribute:
1. Fork this repository.
2. Create a new branch for your feature or bugfix.
3. Write tests for your changes.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This library was inspired by leading research in calibration methods and includes contributions from the open-source community.
