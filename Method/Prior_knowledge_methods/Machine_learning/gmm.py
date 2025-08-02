"""
Gaussian Mixture Model (GMM) for pulse shape discrimination.
Features total charge and integral ratio for neutron/gamma classification.

Reference:
- Gelfusa, M., R. Rossi, M. Lungaroni, F. Belli, L. Spolladore, I. Wyss, 
  P. Gaudio, A. Murari, and J. E. T. Contributors. "Advanced pulse shape 
  discrimination via machine learning for applications in thermonuclear fusion." 
  Nuclear Instruments and Methods in Physics Research Section A: Accelerators, 
  Spectrometers, Detectors and Associated Equipment 974 (2020): 164198.
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os
import torch


# Global model instance
gmm_model = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the GMM model with charge-based features.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global gmm_model
    if task != "classification":
        raise ValueError("GMM only supports classification task.")

    # Convert to numpy arrays
    if torch.is_tensor(train_data):
        train_data = train_data.numpy()
    if torch.is_tensor(train_labels):
        train_labels = train_labels.numpy()

    # Extract features: total charge and integral ratio
    num_train, num_features = train_data.shape
    train_features = np.zeros((num_train, 2))
    for i in range(num_train):
        total_charge = np.sum(train_data[i])
        if num_features >= 80 and total_charge != 0:
            ratio = np.sum(train_data[i][79:]) / total_charge
        else:
            ratio = 0
        train_features[i] = [total_charge, ratio]

    # Train model
    os.makedirs('Output/Trained_models', exist_ok=True)
    gmm_model = GaussianMixture(n_components=2, max_iter=10000, random_state=42)
    gmm_model.fit(train_features)

    # Evaluate performance
    predictions = gmm_model.predict(train_features)
    accuracy = np.mean(predictions == train_labels)
    print(f"Training accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    filename = f"Output/Trained_models/GMM_{task}.joblib"
    joblib.dump(gmm_model, filename)
    print(f"Model saved as '{filename}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, signal_length)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
        
    Returns:
        Predicted class labels (0/1)
    """
    global gmm_model
    if gmm_model is None:
        load_model(task)

    # Extract features: total charge and integral ratio
    num_test, num_features_test = test_data.shape
    test_features = np.zeros((num_test, 2))
    for i in range(num_test):
        total_charge = np.sum(test_data[i])
        if num_features_test >= 80 and total_charge != 0:
            ratio = np.sum(test_data[i][79:]) / total_charge
        else:
            ratio = 0
        test_features[i] = [total_charge, ratio]

    return gmm_model.predict(test_features)


def load_model(task, feat_name=None):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global gmm_model
    if task != "classification":
        raise ValueError("GMM only supports classification task.")

    filename = f"Output/Trained_models/GMM_{task}.joblib"
    gmm_model = joblib.load(filename)
    print("Model loaded successfully.")
