"""
Decision Tree (DT) model for pulse shape discrimination.
Features PCA-based dimensionality reduction and tree-based classification/regression.

When combined with the Scalogram-based Discrimination (SD) feature extractor,
this forms the SDR method proposed by Wolski, D., et al.

Reference:
- Wolski, D., et al. "Comparison of n-γ discrimination by zero-crossing
  and digital charge comparison methods." Nuclear Instruments and
  Methods in Physics Research Section A: Accelerators, Spectrometers,
  Detectors and Associated Equipment 360.3 (1995): 584-592.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib
import os
import torch
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Global model instances
pca = None
dt_model = None


def get_supported_tasks():
    """Return supported tasks (classification/regression)."""
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the DT model with PCA feature extraction.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global pca, dt_model
    
    # Convert to numpy arrays
    if torch.is_tensor(train_data):
        train_data = train_data.numpy()
    if torch.is_tensor(train_labels):
        train_labels = train_labels.numpy()

    # Setup output directory
    os.makedirs('Output/Trained_models', exist_ok=True)

    # Extract features using PCA
    pca = PCA(n_components=2)  # Reduce to 2D feature space
    features = pca.fit_transform(train_data)

    # Initialize and train model
    if task == "classification":
        dt_model = DecisionTreeClassifier(random_state=42)
    else:  # regression
        dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(features, train_labels)

    # Evaluate performance
    train_pred = dt_model.predict(features)
    if task == "classification":
        accuracy = np.mean(train_pred == train_labels)
        print(f"Training accuracy: {accuracy * 100:.2f}%")
    else:
        miu, sigma, fom = histogram_fitting_compute_fom(train_pred, 'DT_train', show_plot=True)
        print(f"Training FOM: {fom:.4f}")

    # Save models
    model_dict = {'pca': pca, 'dt_model': dt_model}
    filename = f"Output/Trained_models/DT_{task}.joblib" if task == "classification" else f"Output/Trained_models/DT_{task}_{feat_name}.joblib"
    joblib.dump(model_dict, filename)
    print(f"Model saved as '{filename}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, signal_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global pca, dt_model
    if pca is None or dt_model is None:
        load_model(task, feat_name)
        
    # Convert and process data
    if torch.is_tensor(test_data):
        test_data = test_data.numpy()
    features = pca.transform(test_data)
    return dt_model.predict(features)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global pca, dt_model
    filename = f"Output/Trained_models/DT_{task}.joblib" if task == "classification" else f"Output/Trained_models/DT_{task}_{feat_name}.joblib"
    
    # Load models
    model_dict = joblib.load(filename)
    pca = model_dict['pca']
    dt_model = model_dict['dt_model']
    print("Model loaded successfully.")