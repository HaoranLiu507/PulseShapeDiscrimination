"""
K-Nearest Neighbors (KNN) model for pulse shape discrimination.
Features segment-based feature extraction and distance-based classification/regression.

Reference:
- Durbin, Matthew, M. A. Wonders, Marek Flaska, and Azaree T. Lintereur.
  "K-nearest neighbors regression for the discrimination of gamma rays and
  neutrons in organic scintillators." Nuclear Instruments and Methods in Physics
  Research Section A: Accelerators, Spectrometers, Detectors and Associated
  Equipment 987 (2021): 164826.
"""
import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Global model instance
knn_model = None


def get_supported_tasks():
    """Return supported tasks (classification/regression)."""
    return ["classification", "regression"]


def compute_features(pulse_signal, num_segments=10):
    """
    Compute segment-based features from a pulse signal.
    
    Args:
        pulse_signal: Input pulse signal (1D array)
        num_segments: Number of segments to divide signal into
        
    Returns:
        Feature vector (sum of each segment)
    """
    pulse_length = len(pulse_signal)
    segment_length = pulse_length // num_segments
    feature_vector = []

    for i in range(num_segments):
        start = i * segment_length
        if i == num_segments - 1:
            segment = pulse_signal[start:]
        else:
            segment = pulse_signal[start:start + segment_length]
        feature_vector.append(np.sum(segment))

    return np.array(feature_vector)


def extract_features(signal_data, num_segments=10):
    """
    Extract features from multiple pulse signals.
    
    Args:
        signal_data: Array of pulse signals (N, signal_length)
        num_segments: Number of segments per signal
        
    Returns:
        Feature matrix (N, num_segments)
    """
    features = [compute_features(pulse, num_segments) for pulse in signal_data]
    return np.vstack(features)


def train(training_data, training_targets, task, feat_name):
    """
    Train the KNN model with segment-based features.
    
    Args:
        training_data: Input signals (N, signal_length)
        training_targets: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global knn_model
    
    # Extract features
    X_train = extract_features(training_data)
    
    # Initialize model
    if task == "regression":
        knn_model = KNeighborsRegressor(n_neighbors=5)
    elif task == "classification":
        knn_model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Train model
    knn_model.fit(X_train, training_targets)

    # Evaluate performance
    predictions = knn_model.predict(X_train)
    if task == "classification":
        accuracy = accuracy_score(training_targets, predictions)
        print(f"Training accuracy: {accuracy * 100:.2f}%")
    else:
        mse = mean_squared_error(training_targets, predictions)
        print(f"Training MSE: {mse:.4f}")
        _, _, fom = histogram_fitting_compute_fom(predictions, "KNN_train", show_plot=True)
        print(f"Training FOM: {fom:.4f}")

    # Save model
    os.makedirs("Output/Trained_models", exist_ok=True)
    model_filename = f"Output/Trained_models/KNN_{task}_{feat_name}.pkl" if task == "regression" else "Output/Trained_models/KNN_classification.pkl"
    joblib.dump(knn_model, model_filename)
    print(f"Model saved as '{model_filename}'")


def test(testing_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        testing_data: Input signals (N, signal_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        
    Returns:
        Predicted class labels (0/1) or continuous values
    """
    global knn_model
    if knn_model is None:
        load_model(task, feat_name)

    X_test = extract_features(testing_data)
    return knn_model.predict(X_test)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global knn_model
    model_filename = f"Output/Trained_models/KNN_{task}_{feat_name}.pkl" if task == "regression" else "Output/Trained_models/KNN_classification.pkl"

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file '{model_filename}' not found.")

    knn_model = joblib.load(model_filename)
    print("Model loaded successfully.")
