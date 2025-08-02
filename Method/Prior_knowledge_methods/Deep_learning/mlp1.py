"""
Single Layer Perceptron for pulse shape discrimination.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from Utility.histogram_fitting_compute_fom import histogram_fitting_compute_fom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
model = None


class MLP1(nn.Module):
    def __init__(self, input_length):
        super(MLP1, self).__init__()
        self.fc = nn.Linear(input_length, 1)

    def forward(self, x):
        # Linear layer with sigmoid activation
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


def get_supported_tasks():
    return ["classification", "regression"]


def train(train_data, train_labels, task, feat_name):
    """
    Train the MLP1 model.
    
    Args:
        train_data: Input data tensor (N, input_length)
        train_labels: Target labels (0/1 for classification, PSD factors for regression)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    """
    global model
    # Convert inputs to tensors if needed
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

    # Setup model and training
    os.makedirs('Output/Trained_models', exist_ok=True)
    input_length = train_data.shape[1]
    model = MLP1(input_length).to(device)
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
        print()
        print('Evaluating on training set...')
        output = model(train_data).squeeze().cpu()

        if task == "classification":
            predictions = torch.round(output)
            accuracy = (predictions == train_labels.cpu()).sum().item() / train_labels.shape[0]
            print("Training accuracy: {:.2f}%".format(accuracy * 100))
        else:
            miu, sigma, fom = histogram_fitting_compute_fom(output, 'MLP1_train', show_plot=True)
            print(f"Training PSD factors computed. FOM: {fom}")

    # Save model
    model_path = f"Output/Trained_models/MLP1_{task}_{feat_name}.pth" if task != "classification" else f"Output/Trained_models/MLP1_{task}.pth"
    torch.save(model.state_dict(), model_path)
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
        test_data: Input data tensor (N, input_length)
        task: Task type (classification/regression)
        feat_name: Feature extractor name for regression
    
    Returns:
        Predictions as numpy array (0/1 for classification, continuous for regression)
    """
    global model
    if model is None:
        load_model(task, feat_name)
    if not torch.is_tensor(test_data):
        test_data = torch.tensor(test_data, dtype=torch.float32)

    test_data = test_data.to(device)

    if test_data.shape[1] != model.fc.in_features:
        raise ValueError(f"Input length mismatch: model expects input length {test_data.shape[1]}, but got {model.fc.in_features}")

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
    model_path = f"Output/Trained_models/MLP1_{task}.pth" if task == "classification" else f"Output/Trained_models/MLP1_{task}_{feat_name}.pth"
    state_dict = torch.load(model_path, map_location=device)
    input_length = state_dict["fc.weight"].shape[1]
    model = MLP1(input_length).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
