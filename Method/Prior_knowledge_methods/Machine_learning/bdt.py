"""
Boosted Decision Tree (BDT) model for pulse shape discrimination.
Features ensemble of weak decision trees using AdaBoost for improved performance.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
import torch
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom


# Global model instance
adaboost_model = None


def get_supported_tasks():
    """Return supported tasks (classification/regression)."""
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the BDT model with AdaBoost.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1 for classification, continuous for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name
    """
    global adaboost_model

    if torch.is_tensor(train_data):
        train_data = train_data.numpy()
    if torch.is_tensor(train_labels):
        train_labels = train_labels.numpy()

    # Setup output directory
    os.makedirs('Output/Trained_models', exist_ok=True)

    # Initialize base estimator
    estimator = DecisionTreeClassifier(max_depth=1) if task == "classification" else DecisionTreeRegressor(max_depth=5)

    # Create AdaBoost model
    if task == "classification":
        adaboost_model = AdaBoostClassifier(estimator=estimator, n_estimators=50, random_state=42)
    elif task == "regression":
        adaboost_model = AdaBoostRegressor(estimator=estimator, n_estimators=50, random_state=42)

    # Train model
    adaboost_model.fit(train_data, train_labels)

    # Evaluate performance
    train_pred = adaboost_model.predict(train_data)
    if task == "classification":
        accuracy = np.mean(train_pred == train_labels)
        print(f"Training accuracy: {accuracy * 100:.2f}%")
    elif task == "regression":
        miu, sigma, fom = histogram_fitting_compute_fom(train_pred, 'BDT_train', show_plot=False)
        print(f"Training FOM: {fom:.4f}")

    # Save model
    if task == "classification":
        filename = f"Output/Trained_models/BDT_{task}.joblib"
    else:
        filename = f"Output/Trained_models/BDT_{task}_{feat_name}.joblib"
    joblib.dump(adaboost_model, filename)
    print(f"Model saved as '{filename}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, signal_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name
        
    Returns:
        Predicted class labels (0/1) or continuous values
    """
    global adaboost_model
    if adaboost_model is None:
        load_model(task, feat_name)

    if torch.is_tensor(test_data):
        test_data = test_data.numpy()

    return adaboost_model.predict(test_data)


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global adaboost_model
    if task == "classification":
        filename = f"Output/Trained_models/BDT_{task}.joblib"
    else:
        filename = f"Output/Trained_models/BDT_{task}_{feat_name}.joblib"
    adaboost_model = joblib.load(filename)
    print("Model loaded successfully.")
