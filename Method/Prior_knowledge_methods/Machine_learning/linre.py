"""
Linear Regression (LINRE) model for pulse shape discrimination.
Features PCA-based dimensionality reduction and linear regression for PSD factor prediction.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import joblib
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Global model instances
pca_model = None
linear_regression_model = None


def get_supported_tasks():
    """Return supported tasks (regression only)."""
    return ["regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the LINRE model with PCA feature extraction.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target PSD factors (N,)
        task: Task type (must be regression)
        feat_name: Feature extractor name
    """
    global pca_model, linear_regression_model
    if task != "regression":
        raise ValueError("LINRE only supports regression task.")

    # Extract features using PCA
    pca_model = PCA(n_components=2)  # Reduce to 2D feature space
    pca_transformed_train = pca_model.fit_transform(train_data)

    # Initialize and train model
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(pca_transformed_train, train_labels)

    # Evaluate performance
    train_predictions = linear_regression_model.predict(pca_transformed_train)
    miu, sigma, fom = histogram_fitting_compute_fom(train_predictions, 'LINRE_train', show_plot=True)
    print(f"Training FOM: {fom:.4f}")

    # Save models
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_dict = {"pca_model": pca_model, "linear_regression_model": linear_regression_model}
    model_path = f"Output/Trained_models/LINRE_{task}_{feat_name}.pkl"
    joblib.dump(model_dict, model_path)
    print(f"Model saved as '{model_path}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, signal_length)
        task: Task type (must be regression)
        feat_name: Feature extractor name
        
    Returns:
        Predicted PSD factors
    """
    global pca_model, linear_regression_model
    if task != "regression":
        raise ValueError("LINRE only supports regression task.")
    if pca_model is None or linear_regression_model is None:
        load_model(task, feat_name)

    # Transform and predict
    pca_transformed_test = pca_model.transform(test_data)
    return linear_regression_model.predict(pca_transformed_test)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (must be regression)
        feat_name: Feature extractor name
    """
    global pca_model, linear_regression_model
    if task != "regression":
        raise ValueError("LINRE only supports regression task.")
        
    # Load models
    model_path = f"Output/Trained_models/LINRE_{task}_{feat_name}.pkl"
    model_dict = joblib.load(model_path)
    pca_model = model_dict["pca_model"]
    linear_regression_model = model_dict["linear_regression_model"]
    print("Model loaded successfully.")
