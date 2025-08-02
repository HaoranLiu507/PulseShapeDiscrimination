"""
Support Vector Machine (SVM) for pulse shape discrimination.
Uses tail-to-total ratio and total charge features with linear kernel SVM.

Reference:
- Sanderson, T. S., C. D. Scott, M. Flaska, J. K. Polack, and S. A. Pozzi. 
  "Machine learning for digital pulse shape discrimination." In 2012 IEEE Nuclear Science 
  Symposium and Medical Imaging Conference Record (NSS/MIC), pp. 199-202. IEEE, 2012.
"""
import numpy as np
from sklearn.svm import SVC
import joblib
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom


# Global model instance
svm_model = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def extract_tail_total_features(DATA, energy_pre=20, energy_post=80, part_start=15):
    """
    Extract tail-to-total ratio and total charge features from pulses.
    
    Args:
        DATA: Input signals (N, signal_length)
        energy_pre: Number of bins before peak for total charge
        energy_post: Number of bins after peak for total charge
        part_start: Starting bin for tail charge calculation
        
    Returns:
        Array of features (N, 2): [tail_ratio, total_charge]
    """
    n_samples, n_bins = DATA.shape
    features = []
    for i in range(n_samples):
        pulse = DATA[i, :]
        peak_idx = np.argmax(pulse)
        # Calculate total charge
        start_idx = max(0, peak_idx - energy_pre)
        end_idx = min(n_bins, peak_idx + energy_post)
        Q_total = np.sum(pulse[start_idx:end_idx])
        # Calculate tail charge
        start_part = peak_idx + part_start
        end_part = min(n_bins, peak_idx + energy_post)
        Q_part = np.sum(pulse[start_part:end_part])
        tail_ratio = Q_part / Q_total if Q_total > 0 else 0
        features.append([tail_ratio, Q_total])
    return np.array(features)


def train(train_data, train_labels, task, feat_name):
    """
    Train the SVM model with tail-to-total features.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global svm_model
    if task != "classification":
        raise ValueError("SVM only supports classification task.")

    # Convert to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Train model
    os.makedirs('Output/Trained_models', exist_ok=True)
    features = extract_tail_total_features(train_data)
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(features, train_labels)

    # Evaluate performance
    train_pred = svm_model.predict(features)
    accuracy = np.mean(train_pred == train_labels)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Save model
    model_dict = {'svm_model': svm_model}
    filename = f"Output/Trained_models/SVM_{task}.joblib"
    joblib.dump(model_dict, filename)
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
    global svm_model
    if svm_model is None:
        load_model(task)

    features = extract_tail_total_features(test_data)
    return svm_model.predict(features)


def load_model(task, feat_name=None):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global svm_model
    if task != "classification":
        raise ValueError("SVM only supports classification task.")

    filename = f"Output/Trained_models/SVM_{task}.joblib"
    model_dict = joblib.load(filename)
    svm_model = model_dict['svm_model']
    print("Model loaded successfully.")
