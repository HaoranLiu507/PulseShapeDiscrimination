"""
Shallow 1D CNN for pulse shape discrimination using raw signals.

Reference:
- Griffiths, Jack, Steven Kleinegesse, D. Saunders, R. Taylor, and
Antonin Vacheret. "Pulse shape discrimination and exploration
of scintillation signals using convolutional neural networks."
Machine Learning: Science and Technology 1, no. 4 (2020): 045022.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom


def signals_to_tensor_1d(signal_list):
    """
    Convert list of 1D signals to tensor with shape (N, 1, signal_length).
    """
    tensor_list = []
    for sig in signal_list:
        sig_array = np.array(sig, dtype=np.float32)
        tensor_list.append(torch.tensor(sig_array).unsqueeze(0))
    return torch.stack(tensor_list)


class CNNShallowModel(nn.Module):
    def __init__(self, task="classification"):
        """
        Initialize shallow 1D CNN model.
        
        Args:
            task: Task type (classification/regression)
        """
        super(CNNShallowModel, self).__init__()
        self.task = task
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=10, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=10, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )
        if task == "classification":
            self.classifier.add_module("fc_out", nn.Linear(64, 2))
        else:
            self.classifier.add_module("fc_out", nn.Linear(64, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Global model instance
_model = None


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_signals, train_labels, task, feat_name, batch_size=256):
    """
    Train the shallow CNN model.
    
    Args:
        train_signals: List of input signals
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Training batch size
    """
    global _model

    # Balance dataset for classification
    if task == "classification":
        train_labels = np.array(train_labels)
        idx0 = np.where(train_labels == 0)[0]
        idx1 = np.where(train_labels == 1)[0]
        min_count = min(len(idx0), len(idx1))
        sample0 = np.random.choice(idx0, min_count, replace=False)
        sample1 = np.random.choice(idx1, min_count, replace=False)
        sel_idx = np.concatenate((sample0, sample1))
        np.random.shuffle(sel_idx)
        train_signals = [train_signals[i] for i in sel_idx]
        train_labels = train_labels[sel_idx]

    # Prepare data
    X_train = signals_to_tensor_1d(train_signals)
    y_train = (torch.tensor(train_labels, dtype=torch.long) if task == "classification"
               else torch.tensor(train_labels, dtype=torch.float32))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and training
    _model = CNNShallowModel(task=task).to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    # Training parameters
    optimizer = optim.Adam(_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    epochs = 50

    # Training loop
    loss_history, epoch_history = [], []
    for epoch in range(epochs):
        _model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = _model(inputs)
            if task == "classification":
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            if task == "classification":
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        avg_loss = epoch_loss / len(train_dataset)
        loss_history.append(avg_loss)
        epoch_history.append(epoch)
        scheduler.step()
        if task == "classification":
            acc = correct / total
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Evaluate model
    _model.eval()
    out_list, target_list = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            outputs = _model(inputs)
            out_list.append(outputs)
            target_list.append(targets)
    outputs = torch.cat(out_list, dim=0)
    targets = torch.cat(target_list, dim=0).to(device)
    if task == "classification":
        _, preds = torch.max(outputs, 1)
        acc = (preds == targets).sum().item() / targets.size(0)
        print("Training accuracy: {:.2f}%".format(acc * 100))
    else:
        targets_np = targets.cpu().detach().numpy()
        miu, sigma, fom = histogram_fitting_compute_fom(targets_np, 'MLP1_train', show_plot=True)
        print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_path = (f"Output/Trained_models/CNNShallow_{task}.pth" if task == "classification"
                  else f"Output/Trained_models/CNNShallow_{task}_{feat_name}.pth")
    torch.save(_model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

    # Plot training loss
    plt.figure()
    plt.plot(epoch_history, loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def test(test_signals, task, feat_name, batch_size=256):
    """
    Test the trained model.
    
    Args:
        test_signals: List of input signals
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Testing batch size
    
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global _model
    if _model is None:
        load_model(task, feat_name)
    X_test = signals_to_tensor_1d(test_signals)
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    _model.eval()
    out_list = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = _model(inputs)
            if task == "classification":
                _, preds = torch.max(outputs, 1)
                out_list.append(preds.cpu())
            else:
                out_list.append(outputs.squeeze().cpu())
    final_output = torch.cat(out_list, dim=0)
    return final_output.numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global _model
    _model = CNNShallowModel(task=task)
    model_path = (f"Output/Trained_models/CNNShallow_{task}.pth" if task == "classification"
                  else f"Output/Trained_models/CNNShallow_{task}_{feat_name}.pth")
    _model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    _model.eval()
    print("Model loaded successfully.")
