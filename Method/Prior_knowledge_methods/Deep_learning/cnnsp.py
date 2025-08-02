"""
CNN for pulse shape discrimination using snapshot-based signal visualization.

Reference:
- Karmakar, Annesha, Anikesh Pal, G. Anil Kumar, and Mohit Tyagi.
  "Neutron-gamma pulse shape discrimination for organic scintillation
  detector using 2D CNN based image classification." Applied Radiation
  and Isotopes 217 (2025): 111653.
"""
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom
from torch.utils.data import TensorDataset, DataLoader


def signal_to_image(signal, img_size=(224, 224)):
    """
    Convert 1D signal to image by drawing signal waveform.
    
    Args:
        signal: 1D array of signal values
        img_size: Output image dimensions (width, height)
    
    Returns:
        PIL Image with drawn signal waveform
    """
    norm_signal = (signal - np.min(signal)) / (np.ptp(signal) + 1e-8)
    width, height = img_size
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)
    L = len(signal)
    xs = np.linspace(0, width - 1, L)
    ys = height - 1 - norm_signal * (height - 1)
    points = list(zip(xs, ys))
    draw.line(points, fill="black", width=1)
    return img


def signals_to_tensor(signals, transform, img_size=(224, 224)):
    """
    Convert list of signals to tensor of waveform images.
    """
    images = []
    import time
    time.sleep(0.1)  # Prevent tqdm from printing ahead of the main process
    for signal in tqdm(signals, desc="Transforming signals to images"):
        img = signal_to_image(signal, img_size)
        img_tensor = transform(img) if transform else transforms.ToTensor()(img)
        images.append(img_tensor)
    return torch.stack(images)


class CNNSPModel(nn.Module):
    def __init__(self, task="classification"):
        """
        Initialize CNN model for snapshot-based signal classification.
        
        Args:
            task: Task type (classification/regression)
        """
        super(CNNSPModel, self).__init__()
        self.task = task
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True)
        )
        if task == "classification":
            self.classifier.add_module("fc_out", nn.Linear(128, 2))
        else:
            self.classifier.add_module("fc_out", nn.Linear(128, 1))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Global model instance
model = None


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name, batch_size=512):
    """
    Train the CNN model.
    
    Args:
        train_data: List of input signals
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Training batch size
    """
    global model
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Balance dataset for classification
    if task == "classification":
        train_labels = np.array(train_labels)
        idx_class0 = np.where(train_labels == 0)[0]
        idx_class1 = np.where(train_labels == 1)[0]
        minority_count = min(len(idx_class0), len(idx_class1))
        idx_class0_sampled = np.random.choice(idx_class0, minority_count, replace=False)
        idx_class1_sampled = np.random.choice(idx_class1, minority_count, replace=False)
        selected_indices = np.concatenate((idx_class0_sampled, idx_class1_sampled))
        np.random.shuffle(selected_indices)
        train_data = [train_data[i] for i in selected_indices]
        train_labels = train_labels[selected_indices]

    # Prepare data
    X_train = signals_to_tensor(train_data, train_transform)
    y_train = torch.tensor(train_labels, dtype=torch.long) if task == "classification" else torch.tensor(train_labels,
                                                                                                         dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNSPModel(task=task).to(device)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    # Training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 15

    # Training loop
    train_loss_list, epoch_list = [], []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
        train_loss_list.append(avg_loss)
        epoch_list.append(epoch)
        if task == "classification":
            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0).to(device)
    if task == "classification":
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == targets).sum().item() / targets.size(0)
        print("Training accuracy: {:.2f}%".format(accuracy * 100))
    else:
        miu, sigma, fom = histogram_fitting_compute_fom(outputs.squeeze().cpu().numpy(),
                                                       'CNNSP_train', show_plot=True)
        print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    os.makedirs('Output/Trained_models', exist_ok=True)
    model_path = f"Output/Trained_models/CNNSP_{task}.pth" if task == "classification" else f"Output/Trained_models/CNNSP_{task}_{feat_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

    # Plot training loss
    plt.figure()
    plt.plot(epoch_list, train_loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def test(test_data, task, feat_name, batch_size=512):
    """
    Test the trained model.
    
    Args:
        test_data: List of input signals
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Testing batch size
    
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global model
    if model is None:
        load_model(task, feat_name)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    X_test = signals_to_tensor(test_data, test_transform)

    # Create dataloader for test data
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            if task == "classification":
                _, preds = torch.max(outputs, 1)
                all_outputs.append(preds.cpu())
            else:
                all_outputs.append(outputs.squeeze().cpu())
    final_outputs = torch.cat(all_outputs, dim=0)
    return final_outputs.numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model = CNNSPModel(task=task)
    model_path = f"Output/Trained_models/CNNSP_{task}.pth" if task == "classification" else f"Output/Trained_models/CNNSP_{task}_{feat_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully.")
