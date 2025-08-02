"""
Deep multilayer perceptron (MLP2) for pulse shape discrimination using PCA features.
Extracts 100 principal components from input signals for universal processing across different signal lengths.
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


class MLP2PCA(nn.Module):
    def __init__(self, n_features):
        """
        Initialize MLP2PCA model with PCA preprocessing.
        
        Args:
            n_features: Length of input signal
        """
        super(MLP2PCA, self).__init__()
        self.n_features = n_features

        # PCA layer: reduces input to 100 components (non-trainable)
        self.pca_layer = nn.Linear(n_features, 100, bias=True)
        self.pca_layer.weight.requires_grad = False
        self.pca_layer.bias.requires_grad = False

        # MLP2 layers
        self.fc1 = nn.Linear(100, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 10)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        """
        Forward pass: PCA -> linear layers -> sigmoid.
        
        Args:
            x: Input tensor (N, n_features)
            
        Returns:
            Output tensor (N, 1)
        """
        if x.shape[1] != self.n_features:
            raise ValueError(f"Input signal length ({x.shape[1]}) does not match model input length ({self.n_features}).")
        out = self.pca_layer(x)
        out = torch.relu(self.fc1(out))
        # out = self.dropout1(out)
        out = torch.relu(self.fc2(out))
        # out = self.dropout2(out)
        out = torch.sigmoid(self.fc3(out))
        return out


def get_supported_tasks():
    """
    Return supported tasks.
    
    Returns:
        List of supported tasks (classification/regression)
    """
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the MLP2PCA model.
    
    Args:
        train_data: Input signals (N, n_features)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    if not torch.is_tensor(train_data):
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # Shuffle training data
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Move tensors to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    # Initialize model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    n_features = train_data.shape[1]
    model = MLP2PCA(n_features).to(device)

    # Compute PCA using SVD
    mean = train_data.mean(dim=0)
    X_centered = train_data - mean
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh.T  # Right singular vectors
    V_top = V[:, :100]  # Top 100 components

    # Set PCA layer parameters (non-trainable)
    model.pca_layer.weight.data = V_top.T  # Shape: (100, n_features)
    model.pca_layer.bias.data = - (mean @ V_top)  # Shape: (100,)

    # Training setup
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
            print(f"Training accuracy: {accuracy * 100:.2f}%")
        else:
            miu, sigma, fom = histogram_fitting_compute_fom(output.cpu(), 'MLP2PCA_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    save_dict = {
        'state_dict': model.state_dict(),
        'n_features': n_features
    }
    model_path = f"Output/Trained_models/MLP2PCA_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2PCA_{task}_{feat_name}.pth"
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
        test_data: Input signals (N, n_features)
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

    with torch.no_grad():
        output = model(test_data).squeeze()
        output = output.cpu()

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
    model_path = f"Output/Trained_models/MLP2PCA_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP2PCA_{task}_{feat_name}.pth"
    save_dict = torch.load(model_path, map_location=device)
    n_features = save_dict['n_features']
    model = MLP2PCA(n_features).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")
