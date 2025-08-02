"""
Linear Regression with Short-Time Fourier Transform (LRSTFT) model for pulse shape discrimination.
Features STFT-based feature extraction, PCA dimensionality reduction, and binary classification.

Reference:
Abdelhakim, Assem, and Ehab Elshazly. "Neutron/gamma pulse shape
discrimination using short-time frequency transform."
Analog Integrated Circuits and Signal Processing 111, no. 3 (2022): 387-402.
"""
import numpy as np
from numpy.fft import fft
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Global model instances
pca_model = None
logistic_regression_model = None
max_feature_length = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def extract_features_from_pulse(pulse_signal, window_length, overlap):
    """
    Extract STFT features from a pulse signal.
    
    Args:
        pulse_signal: Input pulse signal
        window_length: STFT window length
        overlap: Overlap between windows
        
    Returns:
        Array of STFT-based features
    """
    signal_length = len(pulse_signal)
    peak_index = np.argmax(pulse_signal)
    if signal_length - peak_index - window_length - 2 * overlap > 0:
        num_windows = int(np.ceil((signal_length - peak_index - window_length - 2 * overlap) / (window_length - overlap)))
    else:
        num_windows = 1
    feature_vector = []
    for window_index in range(num_windows):
        start_index = peak_index + window_index * (window_length - overlap)
        end_index = start_index + window_length
        if end_index > signal_length:
            segment = np.zeros(window_length)
            available_segment = pulse_signal[start_index:signal_length]
            segment[:len(available_segment)] = available_segment
        else:
            segment = pulse_signal[start_index:end_index]
        fft_result = fft(segment)
        magnitude_0 = np.abs(fft_result[0])
        magnitude_1 = np.abs(fft_result[1])
        feature_value = (magnitude_0 - magnitude_1) / window_length
        feature_vector.append(feature_value)
    return np.array(feature_vector)


def train(train_data, train_labels, task, feat_name):
    """
    Train the LRSTFT model with STFT feature extraction.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1)
        task: Task type (must be classification)
        feat_name: Not used (included for compatibility)
    """
    global max_feature_length, pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LRSTFT only supports classification task.")

    # STFT parameters
    window_length = 10
    overlap = 7

    # Extract features using STFT
    feature_list = [extract_features_from_pulse(pulse, window_length, overlap) for pulse in train_data]
    max_feature_length = max(len(features) for features in feature_list)
    feature_matrix_train = np.array([
        np.pad(features, (0, max_feature_length - len(features)), 'constant') if len(features) < max_feature_length else features[:max_feature_length]
        for features in feature_list
    ])

    # Extract features using PCA
    pca_model = PCA(n_components=2)  # Reduce to 2D feature space
    pca_transformed_train = pca_model.fit_transform(feature_matrix_train)

    # Initialize and train model
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(pca_transformed_train, train_labels)

    # Evaluate performance
    train_predictions = logistic_regression_model.predict(pca_transformed_train)
    training_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Training accuracy: {training_accuracy * 100:.2f}%")

    # Save models
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_dict = {"max_feature_length": max_feature_length, "pca_model": pca_model, "logistic_regression_model": logistic_regression_model}
    model_path = f"Output/Trained_models/LRSTFT_{task}.pkl"
    joblib.dump(model_dict, model_path)
    print(f"Model saved as '{model_path}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, signal_length)
        task: Task type (must be classification)
        feat_name: Not used (included for compatibility)
        
    Returns:
        Predicted class labels (0/1)
    """
    global max_feature_length, pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LRSTFT only supports classification task.")
    if max_feature_length is None or pca_model is None or logistic_regression_model is None:
        load_model(task, feat_name)

    # STFT parameters
    window_length = 10
    overlap = 7

    # Extract features using STFT
    feature_list_test = [extract_features_from_pulse(pulse, window_length, overlap) for pulse in test_data]
    feature_matrix_test = np.array([
        np.pad(features, (0, max_feature_length - len(features)), 'constant') if len(features) < max_feature_length else features[:max_feature_length]
        for features in feature_list_test
    ])

    # Transform and predict
    pca_transformed_test = pca_model.transform(feature_matrix_test)
    return logistic_regression_model.predict(pca_transformed_test)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (must be classification)
        feat_name: Not used (included for compatibility)
    """
    global max_feature_length, pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LRSTFT only supports classification task.")
        
    # Load models
    model_path = f"Output/Trained_models/LRSTFT_{task}.pkl"
    model_dict = joblib.load(model_path)
    max_feature_length = model_dict["max_feature_length"]
    pca_model = model_dict["pca_model"]
    logistic_regression_model = model_dict["logistic_regression_model"]
    print("Model loaded successfully.")