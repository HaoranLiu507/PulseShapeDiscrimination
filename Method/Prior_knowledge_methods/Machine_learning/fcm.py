"""
Fuzzy C-Means (FCM) for pulse shape discrimination.
Uses fuzzy clustering to classify neutron/gamma signals based on pulse shape.

Reference:
- Luo, Xiaoliang, Guofu Liu, and Jun Yang. "Neutron/gamma discrimination utilizing fuzzy c-means 
  clustering of the signal from the liquid scintillator." 2010 First International Conference on 
  Pervasive Computing, Signal Processing and Applications. IEEE, 2010.
"""
import numpy as np
import skfuzzy as fuzz
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom


# Global model components
scaler = None  # Data normalizer
cntr = None    # Cluster centers
fcm_model = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def preprocess_data(DATA, feature_n=30):
    """
    Extract fixed-length features from pulse peaks.
    
    Args:
        DATA: Input signals (N, signal_length)
        feature_n: Number of samples to extract after peak
        
    Returns:
        Processed signals (N, feature_n)
    """
    n = DATA.shape[0]
    DATA_processed = np.zeros((n, feature_n))
    for i in range(n):
        pulse = DATA[i, :]
        peak_index = np.argmax(pulse)
        extracted = pulse[peak_index:peak_index + feature_n]
        if len(extracted) < feature_n:
            extracted = np.pad(extracted, (0, feature_n - len(extracted)), mode='constant')
        DATA_processed[i, :] = extracted
    return DATA_processed


def train(train_data, train_labels, task, feat_name):
    """
    Train the FCM model with normalized pulse features.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global scaler, cntr, fcm_model
    if task != "classification":
        raise ValueError("FCM only supports classification task.")

    # Train model
    os.makedirs('Output/Trained_models', exist_ok=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_norm = scaler.fit_transform(train_data)
    train_proc = preprocess_data(train_norm, feature_n=50)

    # FCM clustering
    train_T = train_proc.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        train_T, c=2, m=2.0, error=1e-6, maxiter=10000)

    # Save model
    model_dict = {'scaler': scaler, 'cntr': cntr}
    filename = f"Output/Trained_models/FCM_{task}.joblib"
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
    global scaler, cntr
    if scaler is None or cntr is None:
        load_model(task)

    # Process data
    test_norm = scaler.transform(test_data)
    test_proc = preprocess_data(test_norm, feature_n=50)

    # Compute memberships
    num_test = test_proc.shape[0]
    test_memberships = np.zeros((num_test, 2))
    for i in range(num_test):
        x = test_proc[i, :]
        distances = np.linalg.norm(x - cntr, axis=1)
        if np.any(distances == 0):
            membership = np.zeros(2)
            membership[distances == 0] = 1
        else:
            membership = np.array(
                [1.0 / np.sum((distances[k] / distances) ** (2 / (2 - 1))) for k in range(2)])
        test_memberships[i, :] = membership

    return (test_memberships[:, 1] > 0.5).astype(int)


def load_model(task, feat_name=None):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global scaler, cntr
    if task != "classification":
        raise ValueError("FCM only supports classification task.")

    filename = f"Output/Trained_models/FCM_{task}.joblib"
    model_dict = joblib.load(filename)
    scaler = model_dict['scaler']
    cntr = model_dict['cntr']
    print("Model loaded successfully.")
