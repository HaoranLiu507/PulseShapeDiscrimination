"""
Logistic Regression (LOGRE) model for pulse shape discrimination.
Features PCA-based dimensionality reduction and binary classification.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Global model instances
pca_model = None
logistic_regression_model = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the LOGRE model with PCA feature extraction.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1)
        task: Task type (must be classification)
        feat_name: Not used (included for compatibility)
    """
    global pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LOGRE only supports classification task.")

    # Extract features using PCA
    pca_model = PCA(n_components=2)  # Reduce to 2D feature space
    pca_transformed_train = pca_model.fit_transform(train_data)

    # Initialize and train model
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(pca_transformed_train, train_labels)

    # Evaluate performance
    train_predictions = logistic_regression_model.predict(pca_transformed_train)
    training_accuracy = accuracy_score(train_labels, train_predictions)
    print(f"Training accuracy: {training_accuracy * 100:.2f}%")

    # Save models
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_dict = {"pca_model": pca_model, "logistic_regression_model": logistic_regression_model}
    model_path = f"Output/Trained_models/LR_{task}.pkl"
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
    global pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LOGRE only supports classification task.")
    if pca_model is None or logistic_regression_model is None:
        load_model(task, feat_name)

    # Transform and predict
    pca_transformed_test = pca_model.transform(test_data)
    return logistic_regression_model.predict(pca_transformed_test)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (must be classification)
        feat_name: Not used (included for compatibility)
    """
    global pca_model, logistic_regression_model
    if task != "classification":
        raise ValueError("LOGRE only supports classification task.")
        
    # Load models
    model_path = f"Output/Trained_models/LR_{task}.pkl"
    model_dict = joblib.load(model_path)
    pca_model = model_dict["pca_model"]
    logistic_regression_model = model_dict["logistic_regression_model"]
    print("Model loaded successfully.")
