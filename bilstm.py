# =============================================================================
# Library
# =============================================================================

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# =============================================================================
# Plots
# =============================================================================
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
# Dataset
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
# Model Architecture
# =============================================================================

class sgRNAbiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(sgRNAbiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Define the biLSTM layer
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Define the fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        # Pass the one-hot encoded sequences directly to the biLSTM layer
        output, _ = self.bilstm(x)

        # Reshape the output to pass it through the fully connected layers
        output = output.contiguous().view(-1, self.hidden_size * 2)

        # Pass the output through the fully connected layers
        output = self.dropout(torch.relu(self.fc1(output)))
        output = self.dropout(torch.relu(self.fc2(output)))
        output = self.fc3(output)

        return output

# =============================================================================
# Main Function
# =============================================================================

if __name__ == "__main__":
   # Use CUDA if available, else use CPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Define the sequence length (gRNA_Seq length + PAM length)
   sequence_length = 20 + 3
   batch_size = 200

   # Load the dataset and preprocess the sequences
   df = pd.read_csv('/Data/Daqi_ang_nature_2019_.csv')
   df['combined_seq'] = df['gRNA_Seq'] + df['PAM']
   labels = df['SpCas9-HF1_Efficiency'].values

   # Split the data into train and validation sets with a 70:30 ratio
   train_indices, val_indices = train_test_split(range(len(df)), test_size=0.3, random_state=42)

   # Create training and validation datasets
   train_dataset = sgRNADataset(df['combined_seq'].iloc[train_indices].values, labels[train_indices])
   val_dataset = sgRNADataset(df['combined_seq'].iloc[val_indices].values, labels[val_indices])

   # Create data loaders for training and validation datasets
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

   # Initialize the model and move it to the appropriate device
   model = sgRNAbiLSTM(input_size=92, hidden_size=128, num_layers=2, dropout_rate=0.2).to(device)

   # Define the loss function and optimizer
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Lists to store training and validation losses, and validation correlation scores
   train_losses = []
   val_losses = []
   val_corr_list = []

   # Training and validation loop
   epochs = 20
   for epoch in range(epochs):
       model.train()
       train_loss = 0
       all_train_labels = []
       all_train_preds = []

       # Training loop
       for batch_sequences, batch_labels in train_loader:
           batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
           optimizer.zero_grad()
           outputs = model(batch_sequences)
           loss = criterion(outputs.squeeze(), batch_labels)
           loss.backward()
           optimizer.step()
           train_loss += loss.item()
           all_train_preds.extend(outputs.squeeze().detach().cpu().numpy())
           all_train_labels.extend(batch_labels.detach().cpu().numpy())

       train_loss /= len(train_loader.dataset)
       train_losses.append(train_loss)

       # Validation loop
       model.eval()
       val_loss = 0
       all_val_labels = []
       all_val_preds = []
       with torch.no_grad():
           for batch_sequences, batch_labels in val_loader:
               batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
               outputs = model(batch_sequences.float())
               loss = criterion(outputs.squeeze(), batch_labels)
               val_loss += loss.item()
               all_val_preds.extend(outputs.squeeze().detach().cpu().numpy())
               all_val_labels.extend(batch_labels.detach().cpu().numpy())

           val_loss /= len(val_loader.dataset)
           val_losses.append(val_loss)

           # Compute Spearman's correlation for validation data
           val_spearman_corr, _ = spearmanr(all_val_labels, all_val_preds)
           val_corr_list.append(val_spearman_corr)

           # Print average losses and Spearman's correlation
           print(f'Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val Spearman: {val_spearman_corr:.4f}')

   # Plot training and validation losses
   plot_loss_vs_epochs(train_losses, val_losses)

   # Plot validation correlation scores
   plot_cor_vs_epochs(val_corr_list)

   # Plot the distribution of predicted and actual efficiency values
   plt.figure(figsize=(12, 6))
   sns.histplot(all_val_preds, color='blue', label='Predicted', kde=True)
   sns.histplot(all_val_labels, color='red', label='Actual', kde=True)
   plt.xlabel('Efficiency Values')
   plt.ylabel('Density')
   plt.title('Distribution of Predicted and Actual Efficiency Values')
   plt.legend()
   plt.savefig("Distribution.png")