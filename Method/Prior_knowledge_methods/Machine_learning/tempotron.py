"""
Tempotron model for pulse shape discrimination.
Features spiking neural network with temporal coding.

Reference:
- Liu, Haoran, Peng Li, Mingzhe Liu, Kaimin Wang, Zhuo Zuo, and Bingqi Liu. 
  "Pulse shape discrimination based on the Tempotron: a powerful classifier on GPU." 
  IEEE Transactions on Nuclear Science (2024).
"""
import numpy as np
import torch
import os
import argparse
from Utility.Tempotron import Tempotron as Tp
import time

# Model hyperparameters
DENDRITES_NUM = 25
TAU = 8.4
TAU_S = 2.1
A = 1
ECHO = 1
THRESHOLD = 1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
tempotron = None


def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the Tempotron model with balanced classes.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global tempotron

    # Balance classes
    train_labels = np.array(train_labels)
    idx0 = np.where(train_labels == 0)[0]
    idx1 = np.where(train_labels == 1)[0]
    min_count = min(len(idx0), len(idx1))
    sample0 = np.random.choice(idx0, min_count, replace=False)
    sample1 = np.random.choice(idx1, min_count, replace=False)
    sel_idx = np.concatenate((sample0, sample1))
    np.random.shuffle(sel_idx)
    train_data = np.array([train_data[i] for i in sel_idx])
    train_labels = np.array(train_labels[sel_idx])

    # Initialize model
    os.makedirs('Output/Trained_models', exist_ok=True)
    efficacies = torch.rand(DENDRITES_NUM, dtype=torch.float64) - 0.5  # Initial synaptic weights
    tempotron = Tp(0, TAU, TAU_S, efficacies, A=A, dendrites_num=DENDRITES_NUM, echo=ECHO, threshold=THRESHOLD)

    # Training parameters
    learning_rate = [1e-5, 1e-3]  # Learning rate range
    epoch_count = 30              # Number of training epochs
    momentum = "on"               # Enable momentum
    noise_key = "None"           # No noise augmentation
    Gaussian_sigma = 1e-04       # Gaussian noise parameter
    jitter_sigma = 1e-04         # Jitter noise parameter
    adding_missing_prob = 1e-04  # Spike addition/deletion probability

    # Train model
    tempotron.train(
        epoch=epoch_count,
        batchsize=1,
        learning_rate=learning_rate,
        momentum=momentum,
        noise_key=noise_key,
        Gaussian_sigma=Gaussian_sigma,
        jitter_sigma=jitter_sigma,
        adding_missing_prob=adding_missing_prob,
        train_data=train_data,
        train_labels=train_labels
    )

    # Save model
    save_dict = {
        'state_dict': tempotron.efficacies,
        'input_length': train_data.shape[1]
    }
    model_path = f"Output/Trained_models/Tempotron_{task}.pth"
    torch.save(save_dict, model_path)
    print(f"Model saved as '{model_path}'")


def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, input_length)
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
        
    Returns:
        Predicted class labels (0/1)
    """
    global tempotron
    if tempotron is None:
        load_model(task, feat_name)

    # Process test data
    with torch.no_grad():
        test_spike_time = tempotron.encode_data(test_data)  # Convert to spike times
        test_loss, pred = tempotron.test_batch(batchsize=8, spike_time=test_spike_time, data_labels=None)
    return pred.squeeze().numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification only)
        feat_name: Not used (included for compatibility)
    """
    global tempotron
    model_path = f"Output/Trained_models/Tempotron_{task}.pth"
    
    # Load and initialize
    save_dict = torch.load(model_path, map_location=device)
    tempotron = Tp(0, TAU, TAU_S, save_dict['state_dict'], A=A, dendrites_num=DENDRITES_NUM, echo=ECHO, threshold=THRESHOLD)
    print("Model loaded successfully.")
