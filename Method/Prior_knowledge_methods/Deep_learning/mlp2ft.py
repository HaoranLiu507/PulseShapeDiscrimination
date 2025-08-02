"""
Deep multilayer perceptron (MLP2) for pulse shape discrimination using Fourier Transform features.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


class MLP2FT(nn.Module):
    def __init__(self, input_length):
        """
        Initialize MLP2FT model.
        
        Args:
            input_length: Length of input signal (must be even)
        """
        super(MLP2FT, self).__init__()
        if input_length % 2 != 0:
            raise ValueError("Input length must be even.")
        self.input_length = input_length
        # Compute frequency domain length using real FFT
        freq_length = input_length // 2 + 1
        self.fc1 = nn.Linear(freq_length, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 10)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        """
        Forward pass: FFT -> magnitude -> linear layers -> sigmoid.
        
        Args:
            x: Input tensor (N, input_length)
            
        Returns:
            Output tensor (N, 1)
        """
        if x.shape[1] % 2 != 0:
            x = torch.nn.functional.pad(x, (0, 1), mode='constant', value=0)
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length mismatch: model expects input length {self.input_length}, but got {x.shape[1]}")
        # Compute real FFT and magnitude
        x_fft = torch.fft.rfft(x, dim=1)
        x_mag = torch.abs(x_fft)
        # Forward pass through network
        x = torch.relu(self.fc1(x_mag))
        # Dropout layers commented out for regression performance
        # x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the MLP2FT model.
    
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

    # Pad odd-length signals
    if train_data.shape[1] % 2 != 0:
        train_data = torch.nn.functional.pad(train_data, (0, 1), mode='constant', value=0)

    # Shuffle training data
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Initialize model and training
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    model = MLP2FT(input_length).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_loss_list = []
    epoch_list = []
    total_epochs = 2000

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
            miu, sigma, fom = histogram_fitting_compute_fom(output.cpu(), 'MLP2FT_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    save_dict = {
        'state_dict': model.state_dict(),
        'input_length': input_length
    }
    model_path = f"Output/Trained_models/MLP2FT_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2FT_{task}_{feat_name}.pth"
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
    # Pad odd-length signals
    if test_data.shape[1] % 2 != 0:
        test_data = torch.nn.functional.pad(test_data, (0, 1), mode='constant', value=0)
    if test_data.shape[1] != model.input_length:
        raise ValueError(f"Input length mismatch: model expects input length {model.input_length}, but got {test_data.shape[1]}")
    test_data = test_data.to(device)
    with torch.no_grad():
        output = model(test_data).squeeze().cpu()
    if task == "classification":
        output = torch.round(output)
    return output.numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model_path = f"Output/Trained_models/MLP2FT_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2FT_{task}_{feat_name}.pth"
    save_dict = torch.load(model_path, map_location=device)
    input_length = save_dict['input_length']
    model = MLP2FT(input_length).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")
