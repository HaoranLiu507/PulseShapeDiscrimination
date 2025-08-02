"""
Recurrent Neural Network (RNN) for pulse shape discrimination.
Processes sequential data with 5 RNN units and 15 fully connected nodes.
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


class RNN(nn.Module):
    def __init__(self, input_length, hidden_size=5, output_size=1):
        """
        Initialize RNN model.
        
        Args:
            input_length: Length of input sequence
            hidden_size: Number of RNN units (default: 5)
            output_size: Size of output layer (default: 1)
        """
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 15)
        self.output = nn.Linear(15, output_size)
        self.input_length = input_length

    def forward(self, x):
        """
        Forward pass: RNN -> fully connected layers -> sigmoid.
        
        Args:
            x: Input tensor (N, seq_len) or (N, seq_len, 1)
            
        Returns:
            Output tensor (N, 1)
        """
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length mismatch: model expects {self.input_length}, but got {x.shape[1]}.")
        if x.dim() == 2:
            x = x.unsqueeze(2)  # Add channel dimension
        rnn_out, _ = self.rnn(x)
        fc_out = self.fc(rnn_out[:, -1, :])  # Use last time step
        out = self.output(fc_out)
        out = torch.sigmoid(out)
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
    Train the RNN model.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    if not torch.is_tensor(train_data):
        train_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device)

    # Shuffle training data
    indices = torch.randperm(train_data.size(0))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    # Initialize model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    model = RNN(input_length).to(device)
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
            miu, sigma, fom = histogram_fitting_compute_fom(output.cpu(), 'RNN_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    model_path = f"Output/Trained_models/RNN_{task}.pth" if task == "classification" else f"Output/Trained_models/RNN_{task}_{feat_name}.pth"
    save_dict = {
        'state_dict': model.state_dict(),
        'input_length': input_length
    }
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
        test_data = torch.tensor(test_data, dtype=torch.float32, device=device)

    input_length = test_data.shape[1]
    if hasattr(model, 'input_length') and model.input_length != input_length:
        raise ValueError(f"Input length mismatch: model expects {model.input_length}, but got {input_length}")

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
    model_path = f"Output/Trained_models/RNN_{task}.pth" if task == "classification" else f"Output/Trained_models/RNN_{task}_{feat_name}.pth"
    save_dict = torch.load(model_path, map_location=device)
    input_length = save_dict['input_length']
    model = RNN(input_length).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")