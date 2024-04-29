
# =============================================================================
# Library
# =============================================================================

from scipy import stats
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =============================================================================
# Plots
# =============================================================================

# Ensure you have the matplotlib library installed for plotting
def plot_loss_vs_epochs(train_loss, val_loss):
        """
	Plots the loss vs epochs graph for three different loss values.

        Parameters:
        train_loss (list): A list of loss values for the first graph.
        loss_values2 (list): A list of loss values for the second graph.
        loss_values3 (list): A list of loss values for the third graph.
        """
	# Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the three loss vs epochs graphs
        ax.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Train Loss')
        ax.plot(range(1, len(val_loss) + 1), val_loss, marker='s', label='Val loss')
        # Set the labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epochs')
        ax.legend()
        # Show the plot
        plt.savefig("loss.png")


def plot_cor_vs_epochs(cor_values):
        """
	Plots the loss vs epochs graph given a list of loss values.

        Parameters:
        loss_values (list): A list of loss values, where each value corresponds to the loss for a single epoch.
        """
	# Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot the loss vs epochs graph
        ax.plot(range(1, len(cor_values) + 1), cor_values, marker='o', label = 'Correlation score')
        # Set the labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation score')
        ax.set_title('Correlation score vs Epochs')
        # Show the plot
        plt.savefig("cor.png")

# =============================================================================
# Model Architecture
# =============================================================================
class DNABertRegressionModel:
    def __init__(self, model_name="zhihan1996/DNABERT-2-117M", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        self.device = device

    def process_in_batches(self, sequences, batch_size=1000):
        batched_outputs = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=23)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batched_outputs.append(outputs[0].mean(dim=1))
        return torch.cat(batched_outputs, dim=0)

    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df['combined_seq'] = df['gRNA_Seq'] + df['PAM']
        sequences = df['combined_seq'].tolist()
        targets = df.get('SpCas9-HF1_Efficiency', None)
        if targets is not None:
            targets = targets.to_numpy()
        return sequences, targets

    def train_val_split(self, embeddings, targets, test_size=0.2, random_state=42):
        return train_test_split(embeddings, targets, test_size=test_size, random_state=random_state)

    def create_dataloaders(self, X_train, X_val, y_train, y_val, batch_size=1000):
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

class DynamicRegressionModel(nn.Module):
    def __init__(self, input_size=768, hidden_sizes=[400, 200], activation='tanh', dropout_p=0.0):
        super(DynamicRegressionModel, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i]))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =============================================================================
# Training
# =============================================================================
def train_model(train_loader, val_loader, device, hyperparameters):
    lr = hyperparameters['lr']
    hidden_sizes = hyperparameters['hidden_sizes']
    activation = hyperparameters['activation']
    optimizer_name = hyperparameters['optimizer']
    dropout_p = hyperparameters['dropout_p']
    epochs = hyperparameters['epochs']

    model = DynamicRegressionModel(input_size=768, hidden_sizes=hidden_sizes, activation=activation, dropout_p=dropout_p)
    model.to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        all_val_labels = []
        all_val_preds = []
        val_cors = []
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
                all_val_preds.extend(outputs.squeeze().detach().cpu().numpy())
                all_val_labels.extend(targets.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        val_spearman_corr, _ = stats.spearmanr(all_val_labels, all_val_preds)
        val_cors.append(val_spearman_corr)

        # Print training and validation loss for each epoch
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Cor: {val_spearman_corr:.4f}")

    # Plot training and validation loss
    plot_loss_vs_epochs(train_losses, val_losses)
    plot_cor_vs_epochs(val_cors)


# =============================================================================
# Main function
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dnabert_regression_model = DNABertRegressionModel(device=device)

    # Load and preprocess training data
    sequences, targets = dnabert_regression_model.load_and_preprocess_data('/Data/Daqi_ang_nature_2019_.csv')

    if os.path.exists('train_HAMP.npy'):
        embeddings = np.load('train_HAMP.npy')
    else:
        print("Generating new training embeddings.")
        embeddings = dnabert_regression_model.process_in_batches(sequences).cpu().numpy()
        np.save('train_HAMP.npy', embeddings)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    X_train, X_val, y_train, y_val = dnabert_regression_model.train_val_split(embeddings, targets)
    train_loader, val_loader = dnabert_regression_model.create_dataloaders(X_train, X_val, y_train, y_val)

    # Manually set hyperparameters here
    hyperparameters = {
        'lr': 1e-5,  # Example learning rate
        'hidden_sizes': [256, 64, 32],  # Example sizes of hidden layers
        'activation': 'relu',  # Activation function
        'optimizer': 'Adam',  # Optimizer
        'dropout_p': 0.25,  # Dropout rate
        'epochs': 100  # Number of epochs
    }

    model = train_model(train_loader, val_loader, device, hyperparameters)
    #predict_and_save(model, device)
    print(hyperparameters)
