"""
Mamba network for pulse shape discrimination using raw signals.
Features selective state space model with efficient computation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from mamba_ssm import Mamba as MambaModule
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


class Mamba(nn.Module):
    def __init__(self, input_length, task, num_layers=2):
        """
        Initialize Mamba model with 3 Mamba modules.
        
        Args:
            input_length: Length of input signal
            task: Task type, either "classification" or "regression"
            num_layers: Number of Mamba layers (default: 2)
        """
        super(Mamba, self).__init__()
        self.input_length = input_length
        self.task = task
        
        # Model hyperparameters
        self.d_model = 16  # Hidden dimension
        self.d_state = 8   # State space dimension
        self.expand = 2    # Expansion factor
        
        # Network layers
        self.projection = nn.Linear(1, self.d_model)  # Input projection
        
        # Create 3 Mamba modules using ModuleList
        self.mamba_layers = nn.ModuleList([
            MambaModule(
                d_model=self.d_model,
                d_state=self.d_state,
                expand=self.expand
            ) for _ in range(num_layers)
        ])
        
        # Normaliation layer
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(num_layers)])
        
        # Output projection based on task
        if self.task == "classification":
            self.fc = nn.Linear(self.d_model, 2)  # Binary classification: 2 classes
        else:
            self.fc = nn.Linear(self.d_model, 1)  # Regression

    def forward(self, x):
        """
        Forward pass through the network with Mamba modules.
        
        Args:
            x: Input tensor (batch_size, input_length)
            
        Returns:
            Output tensor (batch_size, 1)
        """
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length mismatch: model expects {self.input_length}, but got {x.shape[1]}")
            
        x = x.unsqueeze(-1)  # Add feature dimension (B, L, 1)
        x = torch.relu(self.projection(x))  # Project input (B, L, d_model)
        
        # Pass through Mamba modules sequentially
        for i, mamba_layer in enumerate(self.mamba_layers):
            residual = x
            x = mamba_layer(x)
            x = self.norm_layers[i](x)
            x = torch.relu(x)
            if x.shape == residual.shape:
                x = x + residual
            
        # Final state or Mean pool
        # x = x[:, -1, :]  # Take final state
        x = x.mean(dim=1) # Mean pool
        
        x = self.fc(x)
        return x


def get_supported_tasks():
    """Return supported tasks (classification/regression)."""
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name, batch_size=1024):
    """
    Train the Mamba model.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Training batch size (default: 4096)
    """
    global model

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
    if not torch.is_tensor(train_data):
        train_data = np.array(train_data)
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        if task == "classification":
            train_labels = torch.tensor(train_labels, dtype=torch.long)
        else:
            train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # Shuffle and move to device
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices].to(device)
    train_labels = train_labels[indices].to(device)

    # Initialize model
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    if task == "classification":
        model = Mamba(input_length, task, num_layers=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    else:
        model = Mamba(input_length, task).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

    # Training loop
    train_loss_list = []
    epoch_list = []
    total_epochs = 20

    for epoch in range(total_epochs):
        model.train()
        
        # Process batches
        for i in range(0, train_data.size(0), batch_size):
            optimizer.zero_grad()
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            output = model(batch_data)
            if task == "classification":
                loss = criterion(output, batch_labels)
            else:
                loss = criterion(output, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        train_loss_list.append(loss.item())
        epoch_list.append(epoch)
        print(f'Epoch [{epoch + 1}/{total_epochs}], Loss: {loss.item():.4f}', flush=True)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        print("\nEvaluating on training set...")
        output = model(train_data).squeeze()
        
        if task == "classification":
            predictions = torch.argmax(output, dim=1)
            accuracy = (predictions == train_labels).sum().item() / train_labels.shape[0]
            print(f"Training accuracy: {accuracy * 100:.2f}%")
        else:
            miu, sigma, fom = histogram_fitting_compute_fom(output.cpu(), 'Mamba_train', show_plot=False)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    save_dict = {
        'state_dict': model.state_dict(),
        'input_length': input_length
    }
    model_path = f"Output/Trained_models/Mamba_{task}.pth" if task == "classification" else f"Output/Trained_models/Mamba_{task}_{feat_name}.pth"
    torch.save(save_dict, model_path)
    print(f"Model saved as '{model_path}'")

    # Plot loss curve
    # plt.figure()
    # plt.plot(epoch_list, train_loss_list, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


def test(test_data, task, feat_name, batch_size=2048):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, input_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Testing batch size (default: 4096)
        
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global model
    if model is None:
        load_model(task, feat_name)
        
    # Prepare data
    if not torch.is_tensor(test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data.to(device)
    
    if test_data.shape[1] != model.input_length:
        raise ValueError(f"Input length mismatch: model expects {model.input_length}, got {test_data.shape[1]}")
    
    # Process batches
    model.eval()
    output_list = []
    with torch.no_grad():
        for i in range(0, test_data.size(0), batch_size):
            batch_data = test_data[i:i+batch_size]
            outputs = model(batch_data)
            if task == "classification":
                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                output_list.append(probs.cpu())
            else:
                output_list.append(outputs.squeeze().cpu())
    output = torch.cat(output_list, dim=0)
    return output.numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model_path = f"Output/Trained_models/Mamba_{task}.pth" if task == "classification" else f"Output/Trained_models/Mamba_{task}_{feat_name}.pth"
    
    # Load and initialize
    save_dict = torch.load(model_path, map_location=device)
    input_length = save_dict['input_length']
    if task == "classification":
        model = Mamba(input_length, task, num_layers=1).to(device)
    else:
        model = Mamba(input_length, task).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")
