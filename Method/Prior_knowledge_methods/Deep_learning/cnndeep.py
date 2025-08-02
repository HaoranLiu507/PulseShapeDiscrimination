"""
Deep CNN for pulse shape discrimination using wavelet transform spectrograms.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import pywt


def compute_spectrogram(signal, target_size=(224, 224)):
    """
    Convert 1D signal to scalogram using Continuous Wavelet Transform.
    Returns normalized RGB image of specified size.
    """
    scales = np.arange(1, 128)
    wavelet = 'morl'
    coefficients, _ = pywt.cwt(signal, scales, wavelet)
    magnitude = np.abs(coefficients)
    log_magnitude = np.log(magnitude + 1e-8)
    norm_spec = (log_magnitude - np.min(log_magnitude)) / (np.ptp(log_magnitude) + 1e-8)
    spec_array = (norm_spec * 255).astype(np.uint8)
    pil_img = Image.fromarray(spec_array).convert("RGB")
    pil_img = pil_img.resize(target_size)
    return pil_img


def signals_to_tensor(signal_list, transform, target_size=(224, 224)):
    """
    Convert list of signals to tensor of spectrogram images.
    """
    tensor_list = []
    import time
    time.sleep(0.1)
    for sig in tqdm(signal_list, desc="Converting signals to scalogram tensors"):
        img = compute_spectrogram(sig, target_size)
        img_tensor = transform(img) if transform else transforms.ToTensor()(img)
        tensor_list.append(img_tensor)
    return torch.stack(tensor_list)


class CNNDeepModel(nn.Module):
    def __init__(self, task="classification"):
        super(CNNDeepModel, self).__init__()
        self.task = task
        self.features = nn.Sequential(
            # Block 1: 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # Final feature map: 14x14 (224/2^4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )
        if task == "classification":
            self.classifier.add_module("fc_out", nn.Linear(128, 2))
        else:
            self.classifier.add_module("fc_out", nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


_model = None


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_signals, train_labels, task, feat_name, batch_size=256):
    """
    Train the deep CNN model.
    
    Args:
        train_signals: List of input signals
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Training batch size
    """
    global _model
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

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
    X_train = signals_to_tensor(train_signals, transform_pipeline)
    y_train = (torch.tensor(train_labels, dtype=torch.long) if task == "classification"
               else torch.tensor(train_labels, dtype=torch.float32))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and training
    _model = CNNDeepModel(task=task).to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    # Training parameters
    if task == "classification":
        optimizer = optim.Adam(_model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        epochs = 500
    else:
        optimizer = optim.Adam(_model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        epochs = 100

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
            loss = (criterion(outputs, targets) if task == "classification"
                    else criterion(outputs.squeeze(), targets))
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
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Evaluate model
    _model.eval()
    out_list, target_list = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            out_list.append(_model(inputs))
            target_list.append(targets)
    outputs = torch.cat(out_list, dim=0)
    targets = torch.cat(target_list, dim=0).to(device)
    if task == "classification":
        _, preds = torch.max(outputs, 1)
        acc = (preds == targets).sum().item() / targets.size(0)
        print("Training accuracy: {:.2f}%".format(acc * 100))
    else:
        mu, sigma, fom = histogram_fitting_compute_fom(outputs.squeeze().cpu().numpy(),
                                                       'CNNDeep_train', show_plot=True)
        print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_path = (f"Output/Trained_models/CNNDeep_{task}.pth" if task == "classification"
                  else f"Output/Trained_models/CNNDeep_{task}_{feat_name}.pth")
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
        Predictions as numpy array (0/1 for classification, continuous for regression)
    """
    global _model
    if _model is None:
        load_model(task, feat_name)
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    X_test = signals_to_tensor(test_signals, transform_pipeline)
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
    _model = CNNDeepModel(task=task)
    model_path = (f"Output/Trained_models/CNNDeep_{task}.pth" if task == "classification"
                  else f"Output/Trained_models/CNNDeep_{task}_{feat_name}.pth")
    _model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    _model.eval()
    print("Model loaded successfully.")
