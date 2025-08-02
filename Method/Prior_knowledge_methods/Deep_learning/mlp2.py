"""
Deep multilayer perceptron (MLP2) for pulse shape discrimination.
Also called the dense neural network (DNN) in the reference.

Reference:
- Dutta, Shubham, Sayan Ghosh, Satyaki Bhattacharya, and Satyajit Saha.
  "Pulse shape simulation and discrimination using machine learning techniques."
  Journal of Instrumentation 18, no. 03 (2023): P03038.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


def extract_features(pulses, trunc_points=None):
    """
    Calculate normalized pulse areas at different truncation points.
    
    Args:
        pulses: Input signals (n_signals, signal_length)
        trunc_points: List of truncation points. Default: 6 evenly spaced points.
    
    Returns:
        Normalized features (n_signals, 6)
    """
    n_signals, signal_length = pulses.shape
    if trunc_points is None:
        trunc_points = [int(signal_length * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    features = np.zeros((n_signals, len(trunc_points)))
    for i in range(n_signals):
        pulse = pulses[i, :]
        total_area = np.sum(pulse)
        if total_area < 1e-9:
            features[i, :] = 0
        else:
            cum_area = np.cumsum(pulse)
            features[i, :] = [cum_area[t] / total_area for t in trunc_points]
    return features


class MLP2(nn.Module):
    def __init__(self, input_dim=6, task="classification"):
        """
        Initialize MLP2 model.
        
        Args:
            input_dim: Input feature dimension (default: 6)
            task: Task type (classification/regression)
        """
        super(MLP2, self).__init__()
        self.task = task
        self.fc1 = nn.Linear(input_dim, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 10)
        self.dropout2 = nn.Dropout(0.5)
        if self.task == "classification":
            self.fc3 = nn.Linear(10, 2)
        else:  # regression
            self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        if self.task == "classification":
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the MLP2 model.
    
    Args:
        train_data: Input signals (N, signal_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model

    # Extract features
    if not isinstance(train_data, np.ndarray):
        raise ValueError("train_data must be a numpy array.")
    features = extract_features(train_data)
    train_data = torch.tensor(features, dtype=torch.float32)

    # Prepare labels
    if task == "classification":
        if not torch.is_tensor(train_labels):
            train_labels = torch.tensor(train_labels, dtype=torch.long)
        target = torch.nn.functional.one_hot(train_labels, num_classes=2).float()
    elif task == "regression":
        if not torch.is_tensor(train_labels):
            train_labels = torch.tensor(train_labels, dtype=torch.float32)
        target = train_labels.unsqueeze(1)
    else:
        print("Error: Task not supported.")
        return

    # Move data to device and shuffle
    train_data = train_data.to(device)
    target = target.to(device)
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    target = target[indices]

    # Initialize model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    model = MLP2(input_dim=6, task=task).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_loss_list = []
    epoch_list = []
    total_epochs = 1000

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
        epoch_list.append(epoch)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item():.4f}', flush=True)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        output = model(train_data)
        if task == "classification":
            predictions = torch.argmax(output, dim=1)
            true_labels = torch.argmax(target, dim=1)
            accuracy = (predictions == true_labels).sum().item() / true_labels.shape[0]
            print("Training accuracy: {:.2f}%".format(accuracy * 100))
        else:  # regression
            output_reg = output.cpu().squeeze()
            miu, sigma, fom = histogram_fitting_compute_fom(output_reg, 'MLP2_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    model_path = f"Output/Trained_models/MLP2_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2_{task}_{feat_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

    # Plot training loss
    plt.figure()
    plt.plot(epoch_list, train_loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


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
    global model
    if model is None:
        load_model(task, feat_name)
    if not isinstance(test_data, np.ndarray):
        raise ValueError("test_data must be a numpy array.")
    features = extract_features(test_data)
    test_data = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(test_data)
    if task == "classification":
        predictions = torch.argmax(output, dim=1).cpu()
        return predictions.numpy()
    else:
        return output.cpu().squeeze().numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model = MLP2(input_dim=6, task=task).to(device)
    model_path = f"Output/Trained_models/MLP2_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2_{task}_{feat_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
