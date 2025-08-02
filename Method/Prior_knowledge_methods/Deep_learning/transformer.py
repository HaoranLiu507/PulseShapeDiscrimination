"""
Transformer network for pulse shape discrimination using attention mechanisms.
Features CLS token aggregation, positional encoding, and multi-head self-attention.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


class Transformer(nn.Module):
    def __init__(self, input_length, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        """
        Initialize Transformer model.
        
        Args:
            input_length: Length of input sequence
            embed_dim: Embedding dimension (default: 64)
            num_layers: Number of encoder layers (default: 2)
            num_heads: Number of attention heads (default: 2)
            dropout: Dropout rate (default: 0.1)
        """
        super(Transformer, self).__init__()
        self.input_length = input_length
        self.embed_dim = embed_dim

        # Project scalar inputs to embedding space
        self.embedding = nn.Linear(1, embed_dim)

        # CLS token for sequence aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        pe = self._generate_positional_encoding(input_length + 1, embed_dim)
        self.register_buffer('positional_encoding', pe)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_layer = nn.Linear(embed_dim, 1)

    def _generate_positional_encoding(self, seq_len, embed_dim):
        """
        Generate sinusoidal positional encodings.
        
        Args:
            seq_len: Length of sequence including CLS token
            embed_dim: Embedding dimension
            
        Returns:
            Positional encoding tensor (seq_len, embed_dim)
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Forward pass: embed -> add CLS & position -> transformer -> CLS token output.
        
        Args:
            x: Input tensor (N, input_length)
            
        Returns:
            Output tensor (N, 1)
        """
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length mismatch: expected {self.input_length}, got {x.shape[1]}")

        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        cls_output = x[:, 0, :]
        output = self.output_layer(cls_output)  # Project to output
        return output


def get_supported_tasks():
    """
    Return supported tasks.
    
    Returns:
        List of supported tasks (classification/regression)
    """
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name, batch_size=256):
    """
    Train the Transformer model.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Batch size for training (default: 256)
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
    
    if not torch.is_tensor(train_data):
        train_data = np.array(train_data)
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    # Initialize model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    model = Transformer(input_length).to(device)

    # Training setup
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop with batches
    total_epochs = 100
    train_loss_list = []
    epoch_list = []
    num_batches = int(np.ceil(train_data.size(0) / batch_size))

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, train_data.size(0))
            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            # Forward and backward pass
            optimizer.zero_grad()
            output = model(batch_data).squeeze()
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log progress
        avg_epoch_loss = epoch_loss / num_batches
        train_loss_list.append(avg_epoch_loss)
        epoch_list.append(epoch)
        print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_epoch_loss:.4f}", flush=True)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        print("\nEvaluating on training set...")
        predictions = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, train_data.size(0))
            batch_data = train_data[start_idx:end_idx]
            output = model(batch_data).squeeze()
            predictions.append(output)
        predictions = torch.cat(predictions, dim=0)
        
        if task == "classification":
            preds = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (preds == train_labels).float().mean().item()
            print(f"Training accuracy: {accuracy * 100:.2f}%")
        else:
            miu, sigma, fom = histogram_fitting_compute_fom(predictions.cpu(), 'Transformer_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    save_dict = {
        'state_dict': model.state_dict(),
        'input_length': input_length
    }
    model_path = f"Output/Trained_models/Transformer_{task}.pth" if task == "classification" else f"Output/Trained_models/Transformer_{task}_{feat_name}.pth"
    torch.save(save_dict, model_path)
    print(f"Model saved as '{model_path}'")

    # Plot training loss
    plt.figure()
    plt.plot(epoch_list, train_loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def test(test_data, task, feat_name, batch_size=256):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, input_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
        batch_size: Batch size for testing (default: 256)
    
    Returns:
        Predictions (0/1 for classification, continuous for regression)
    """
    global model
    if model is None:
        load_model(task, feat_name)
    if not torch.is_tensor(test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data.to(device)

    # Test in batches
    num_batches = int(np.ceil(test_data.size(0) / batch_size))
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, test_data.size(0))
            batch_data = test_data[start_idx:end_idx]
            output = model(batch_data).squeeze()
            predictions.append(output)
        predictions = torch.cat(predictions, dim=0)

    if task == "classification":
        predictions = (torch.sigmoid(predictions) > 0.5).float()

    return predictions.cpu().numpy()


def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    model_path = f"Output/Trained_models/Transformer_{task}.pth" if task == "classification" else f"Output/Trained_models/Transformer_{task}_{feat_name}.pth"
    save_dict = torch.load(model_path, map_location=device)
    input_length = save_dict['input_length']
    model = Transformer(input_length).to(device)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    print("Model loaded successfully.")
