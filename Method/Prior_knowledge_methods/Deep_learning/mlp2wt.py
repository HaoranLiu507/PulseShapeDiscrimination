"""
Deeper multilayer perceptron (MLP2) for pulse shape discrimination using Wavelet Transform features.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pywt
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


class MLP2WT(nn.Module):
    def __init__(self, input_length):
        """
        Initialize MLP2WT model.
        
        Args:
            input_length: Length of input signal (will be padded to even if needed)
        """
        super(MLP2WT, self).__init__()
        # Store original input length and padded length
        self.original_input_length = input_length
        self.padded_input_length = input_length if input_length % 2 == 0 else input_length + 1
        self.fc1 = nn.Linear(self.padded_input_length, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 10)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass: Pad if needed -> DWT -> concatenate coefficients -> linear layers -> sigmoid.
        
        Args:
            x: Input tensor (N, input_length)
            
        Returns:
            Output tensor (N, 1)
        """
        if x.shape[1] != self.original_input_length:
            raise ValueError(
                f"Input signal length ({x.shape[1]}) does not match model input length ({self.original_input_length})."
            )
        
        # Pad with zero if input length is odd
        if self.original_input_length % 2 != 0:
            x = torch.nn.functional.pad(x, (0, 1), mode='constant', value=0)
        
        # Convert tensor to numpy for pywt processing
        x_np = x.cpu().detach().numpy()
        transformed_list = []
        for sample in x_np:
            # Compute one-level DWT using 'db1' wavelet
            cA, cD = pywt.dwt(sample, 'db1')
            # Concatenate approximation and detail coefficients
            transformed_sample = np.concatenate((cA, cD))
            transformed_list.append(transformed_sample)
        transformed_array = np.stack(transformed_list)
        # Convert back to torch tensor
        transformed_tensor = torch.tensor(transformed_array, dtype=torch.float32, device=x.device)
        # Forward pass through network
        out = self.relu(self.fc1(transformed_tensor))
        # Dropout layers commented out for regression performance
        # out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        # out = self.dropout2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the MLP2WT model.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    if not torch.is_tensor(train_data):
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    # Shuffle training data
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Initialize model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    model = MLP2WT(input_length).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    train_loss_list = []
    epoch_list = []
    total_epochs = 1000

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
        epoch_list.append(epoch)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item():.4f}', flush=True)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        print("\nEvaluating on training set...")
        output = model(train_data).squeeze()
        if task == "classification":
            predictions = torch.round(output)
            accuracy = (predictions == train_labels).sum().item() / train_labels.shape[0]
            print("Training accuracy: {:.2f}%".format(accuracy * 100))
        else:
            miu, sigma, fom = histogram_fitting_compute_fom(output.cpu(), 'MLP2WT_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    save_dict = {
        'state_dict': model.state_dict(),
        'input_length': input_length,
        'original_input_length': model.original_input_length,
        'padded_input_length': model.padded_input_length
    }
    model_path = f"Output/Trained_models/MLP2WT_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2WT_{task}_{feat_name}.pth"
    torch.save(save_dict, model_path)
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
        test_data: Input signals (N, input_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global model
    if model is None:
        load_model(task, feat_name)
    if not torch.is_tensor(test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data.to(device)
    if test_data.shape[1] != model.original_input_length:
        raise ValueError(
            f"Input signal length ({test_data.shape[1]}) does not match model input length ({model.original_input_length})."
        )
    with torch.no_grad():
        output = model(test_data).squeeze()
    if task == "classification":
        output = torch.round(output)
    return output.cpu().numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model_path = f"Output/Trained_models/MLP2WT_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2WT_{task}_{feat_name}.pth"
    save_dict = torch.load(model_path, map_location=device)
    input_length = save_dict['input_length']
    model = MLP2WT(input_length).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")
