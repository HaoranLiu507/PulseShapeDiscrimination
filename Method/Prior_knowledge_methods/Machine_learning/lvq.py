"""
Learning Vector Quantization (LVQ) model for pulse shape discrimination.
Features competitive learning with prototype vectors for classification.

Reference:
- Tambouratzis, Tatiana, Dina Chernikova, and Imre Pzsit. "Pulse shape discrimination
  of neutrons and gamma rays using Kohonen artificial neural networks."
  Journal of Artificial Intelligence and Soft Computing Research 3 (2013).
"""
import numpy as np
import torch
import torch.nn as nn
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None

class LVQ(nn.Module):
    def __init__(self, input_length, num_classes, num_codebook_vectors):
        """
        Initialize LVQ model.
        
        Args:
            input_length: Length of input signal
            num_classes: Number of classes (e.g., 2 for neutron/gamma)
            num_codebook_vectors: Number of prototype vectors
        """
        super(LVQ, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.num_codebook_vectors = num_codebook_vectors
        
        # Model parameters
        self.codebook_vectors = nn.Parameter(torch.randn(num_codebook_vectors, input_length))  # Learnable prototypes
        self.codebook_classes = torch.tensor(  # Fixed class assignments
            [i % num_classes for i in range(num_codebook_vectors)], 
            dtype=torch.long
        ).to(device)

    def forward(self, x):
        """
        Forward pass: find nearest prototype and return its class.
        
        Args:
            x: Input tensor (N, input_length)
            
        Returns:
            Predicted classes (N,)
        """
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length mismatch: model expects {self.input_length}, but got {x.shape[1]}")
            
        distances = torch.cdist(x, self.codebook_vectors)  # Compute distances to prototypes
        winners = torch.argmin(distances, dim=1)  # Find nearest prototype
        return self.codebook_classes[winners]  # Return class of nearest prototype

def get_supported_tasks():
    """Return supported tasks (classification only)."""
    return ["classification"]

def train(train_data, train_labels, task, feat_name):
    """
    Train the LVQ model.
    
    Args:
        train_data: Input signals (N, input_length)
        train_labels: Target labels (0/1 for classification)
        task: Task type (classification only)
        feat_name: Feature extractor name (not used)
    """
    global model

    # Balance classes
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

    # Convert to tensors
    if not torch.is_tensor(train_data):
        train_data = torch.tensor(np.array(train_data), dtype=torch.float32)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels, dtype=torch.long)

    # Setup training
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    input_length = train_data.shape[1]
    num_classes = len(torch.unique(train_labels))
    num_codebook_vectors = 100  # Number of prototypes
    model = LVQ(input_length, num_classes, num_codebook_vectors).to(device)

    # Training parameters
    alpha = 0.01  # Learning rate
    total_epochs = 10

    # Training loop
    for epoch in range(total_epochs):
        for i in range(train_data.size(0)):
            x = train_data[i]
            y = train_labels[i]
            
            # Find winning prototype
            distances = torch.cdist(x.unsqueeze(0), model.codebook_vectors)
            winner_idx = torch.argmin(distances, dim=1).item()
            winner_class = model.codebook_classes[winner_idx]
            
            # Update prototype: attract if correct class, repel if wrong
            if winner_class == y:
                model.codebook_vectors.data[winner_idx] += alpha * (x - model.codebook_vectors[winner_idx])
            else:
                model.codebook_vectors.data[winner_idx] -= alpha * (x - model.codebook_vectors[winner_idx])
                
        alpha *= 0.99  # Decay learning rate

        # Evaluate progress
        with torch.no_grad():
            predicted = model(train_data)
            accuracy = (predicted == train_labels).sum().item() / train_labels.size(0)
            print(f'Epoch [{epoch + 1}/{total_epochs}], Accuracy: {accuracy:.4f}')

    # Save model
    os.makedirs('Output/Trained_models', exist_ok=True)
    save_dict = {
        'codebook_vectors': model.codebook_vectors.data,
        'codebook_classes': model.codebook_classes,
        'input_length': input_length,
        'num_classes': num_classes,
        'num_codebook_vectors': num_codebook_vectors
    }
    model_path = f"Output/Trained_models/LVQ_classification.pth"
    torch.save(save_dict, model_path)
    print(f"Model saved as '{model_path}'")

def test(test_data, task, feat_name):
    """
    Test the trained model.
    
    Args:
        test_data: Input signals (N, input_length)
        task: Task type (classification only)
        feat_name: Feature extractor name (not used)
        
    Returns:
        Predictions (array of 0/1)
    """
    global model
    if model is None:
        load_model(task, feat_name)
    if not torch.is_tensor(test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data.to(device)
    
    if test_data.shape[1] != model.input_length:
        raise ValueError(f"Input length mismatch: model expects {model.input_length}, but got {test_data.shape[1]}")
        
    with torch.no_grad():
        predicted = model(test_data)
    return predicted.cpu().numpy()

def load_model(task, feat_name):
    """
    Load a trained model.
    
    Args:
        task: Task type (classification only)
        feat_name: Feature extractor name (not used)
    """
    global model
    model_path = f"Output/Trained_models/LVQ_classification.pth"
    save_dict = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    input_length = save_dict['input_length']
    num_classes = save_dict['num_classes']
    num_codebook_vectors = save_dict['num_codebook_vectors']
    model = LVQ(input_length, num_classes, num_codebook_vectors).to(device)
    model.codebook_vectors.data = save_dict['codebook_vectors']
    model.codebook_classes = save_dict['codebook_classes']
    model.eval()
    print("Model loaded successfully.")