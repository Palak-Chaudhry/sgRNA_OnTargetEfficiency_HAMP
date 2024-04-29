
# =============================================================================
# Library
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =============================================================================
# Plotting
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
        plt.savefig("loss_MLP.png")


def plot_cor_vs_epochs(test_cor, train_cor):
        """
	Plots the loss vs epochs graph given a list of loss values.

        Parameters:
        loss_values (list): A list of loss values, where each value corresponds to the loss for a single epoch.
        """
	# Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot the loss vs epochs graph
        ax.plot(range(1, len(test_cor) + 1), test_cor, marker='o', label = 'Test Correlation score')
        ax.plot(range(1, len(train_cor) + 1), train_cor, marker='s', label=' Train Correlation score')
        # Set the labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation score')
        ax.set_title('Correlation score vs Epochs')
        ax.legend()
        # Show the plot
        plt.savefig("cor_MLP.png")


# =============================================================================
# Data preprocessing
# =============================================================================

class sgRNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.encoder = LabelBinarizer()
        self.encoder.fit(list('ACGT'))  # Assuming only A, C, G, T are present

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_seq = self.one_hot_encode(sequence)
        label = self.labels[idx]
        return torch.tensor(encoded_seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def one_hot_encode(self, sequence):
        encoded = self.encoder.transform(list(sequence)).flatten()

        # If the sequence is shorter than the desired length, pad it with zeros
        if len(sequence) < sequence_length:
            padding = sequence_length - len(sequence)
            encoded = np.pad(encoded, (0, padding * 4), 'constant')

        # If the sequence is longer, truncate it
        encoded = encoded[:sequence_length * 4]
        return encoded

# =============================================================================
# MLP Regression Architecture
# =============================================================================

class sgRNAModel(nn.Module):
    def __init__(self, input_size):
        super(sgRNAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# =============================================================================
# Main functions
# =============================================================================
if __name__ == "__main__":

    # Set the device to 'cuda' if a CUDA-enabled GPU is available, otherwise use 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the sequence length as the sum of gRNA sequence length (20) and PAM length (3)
    sequence_length = 20 + 3

    # Load the data from the CSV file into a pandas DataFrame
    df = pd.read_csv('/Data/Daqi_ang_nature_2019.csv')
    # Concatenate the 'gRNA_Seq' and 'PAM' columns to create a new 'combined_seq' column
    df['combined_seq'] = df['gRNA_Seq'] + df['PAM']
    # Extract the 'SpCas9-HF1_Efficiency' column values as the labels
    labels = df['SpCas9-HF1_Efficiency'].values

    # Split the data into train and temp indices using train_test_split
    train_indices, temp_indices = train_test_split(range(len(df)), test_size=0.4, random_state=42)

    # Split the temp indices into validation and test indices using train_test_split
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.75, random_state=42)

    # Create the training, validation, and test datasets using the custom sgRNADataset class
    train_dataset = sgRNADataset(df['combined_seq'].iloc[train_indices].values, labels[train_indices])
    val_dataset = sgRNADataset(df['combined_seq'].iloc[val_indices].values, labels[val_indices])
    test_dataset = sgRNADataset(df['combined_seq'].iloc[test_indices].values, labels[test_indices])

    # Set the batch size for the data loaders
    batch_size = 200

    # Create the training, validation, and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate the input size for the model as 4 * sequence_length (4 nucleotide types * sequence length)
    input_size = 4 * sequence_length

    # Create an instance of the sgRNAModel and move it to the specified device
    model = sgRNAModel(input_size).to(device)

    # Define the loss function (Mean Squared Error) and the optimizer (Adam with learning rate 0.001)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store training and validation losses and correlations
    train_losses = []
    val_losses = []
    train_cors = []
    val_cors = []

    # Set the number of training epochs
    epochs = 50

    # Training and validation loop
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()

        # Initialize variables for training loss, labels, and predictions
        train_loss = 0
        all_train_labels = []
        all_train_preds = []

        # Training loop
        for batch_sequences, batch_labels in train_loader:
            # Move the batch data to the specified device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_sequences.float())

            # Compute the loss
            loss = criterion(outputs.squeeze(), batch_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

            # Collect the predictions and labels for computing correlation
            all_train_preds.extend(outputs.squeeze().detach().cpu().numpy())
            all_train_labels.extend(batch_labels.detach().cpu().numpy())

        # Compute the average training loss for the epoch
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Compute the Spearman correlation for the training data
        train_spearman_corr, _ = spearmanr(all_train_labels, all_train_preds)
        train_cors.append(train_spearman_corr)

        # Set the model to evaluation mode
        model.eval()

        # Initialize variables for validation loss, labels, and predictions
        val_loss = 0
        all_val_labels = []
        all_val_preds = []

        # Validation loop
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                # Move the batch data to the specified device
                batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)

                # Forward pass
                outputs = model(batch_sequences.float())

                # Compute the loss
                loss = criterion(outputs.squeeze(), batch_labels)

                # Accumulate the validation loss
                val_loss += loss.item()

                # Collect the predictions and labels for computing correlation
                all_val_preds.extend(outputs.squeeze().detach().cpu().numpy())
                all_val_labels.extend(batch_labels.detach().cpu().numpy())

        # Compute the average validation loss for the epoch
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Compute the Spearman correlation for the validation data
        val_spearman_corr, _ = spearmanr(all_val_labels, all_val_preds)
        val_cors.append(val_spearman_corr)

        # Print the average losses and Spearman correlations for the epoch
        print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Train Spearman: {train_spearman_corr:.4f}, '
              f'Val Spearman: {val_spearman_corr:.4f}')

    # Plot the training and validation losses over epochs
    plot_loss_vs_epochs(train_losses, val_losses)

    # Plot the training and validation Spearman correlations over epochs
    plot_cor_vs_epochs(train_cors, val_cors)