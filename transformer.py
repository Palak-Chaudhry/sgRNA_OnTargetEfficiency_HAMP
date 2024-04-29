# =============================================================================
# Libs
# =============================================================================

from collections import Counter
from os.path import exists
import random
import re
import math
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt

##----------------------------------------------
## Preprocessing
##----------------------------------------------

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def load_train_data(path, val_split=False):
  df = pd.read_csv(path)
  df['sgRNA'] = df['gRNA_Seq'] + df['PAM']
  if val_split:
        num_rows_to_select = int(0.2 * len(df))
        val = df.iloc[:num_rows_to_select]
        train = df.iloc[num_rows_to_select:]
        return train, val
  else:
	  return df

# Preprocessing function
def preprocess_data(sequences, vocab):
    # Convert sequences to indices
    seq_indices = []
    for seq in sequences:
        seq_indices.append([vocab[char.upper()] for char in seq])
    max_len = 23
    padded_seq_indices = [seq + [0]*(max_len - len(seq)) for seq in seq_indices]

    return torch.tensor(padded_seq_indices)

##----------------------------------------------
## Generate plots for evaluation
##----------------------------------------------

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
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        
        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
        
        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        #n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
        
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class Transformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, n_embeddings, seq_len, dropout=.1):
        super().__init__()
        
        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, seq_len)
        
        #backbone
        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)
        
        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(embed_size*23, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)                
    
    def forward(self, x):
        x = self.embeddings(x)
        #print(f"Shape of x after embedding: {x.shape}")
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.linear1(x)
        #print(f"Shape of x at end: {x.shape}")
        return x

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len

if __name__ == "__main__":

    train, val = load_train_data("/Data/Daqi_ang_nature_2019_.csv", val_split=True)

    vocab = "ACGT"
    aa_dict = {k: v+1 for v, k in enumerate(vocab)}
    x_train = preprocess_data(train['21mer'], aa_dict)
    x_val = preprocess_data(val['21mer'], aa_dict)
    train_target = torch.tensor(train['SpCas9-HF1_Efficiency'].values.astype(np.float32))
    train_tensor = TensorDataset(x_train, train_target)
    train_loader = DataLoader(dataset = train_tensor, batch_size = 1000, shuffle = True)

    val_target = torch.tensor(val['SpCas9-HF1_Efficiency'].values.astype(np.float32))
    val_tensor = TensorDataset(x_val, val_target)
    val_loader = DataLoader(dataset = val_tensor, batch_size = 1000, shuffle = True)

    # Define model, loss, and optimizer

    print('initializing..')
    batch_size = 500
    seq_len = 23
    embed_size = 8
    inner_ff_size = embed_size * 4
    n_heads = 4
    n_code = 2
    n_vocab = 5
    dropout = 0
    # n_workers=12

    #Initialise variables to store evaluation metrics
    train__loss = []
    test__loss = []
    correlation = []

    #optimizer
    optim_kwargs = {'lr':1e-5, 'weight_decay':1e-4, 'betas':(.9,.999)}
    print('initializing model...')
    model = Transformer(n_code, n_heads, embed_size, inner_ff_size, n_vocab, seq_len, dropout)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.00001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        print("epoch strted")
        model.train()
        train_loss = 0.0
        for sequences, scores in train_loader:
            optimizer.zero_grad()
            sequences = torch.tensor(sequences).to(device)
            #print("sequences: ", sequences)
            scores = torch.tensor(scores).to(device).unsqueeze(-1)
            #print("scores: ", scores)
            outputs = model(sequences)
            #print("outputs: ",outputs.squeeze())
            loss = criterion(outputs.squeeze(), scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sequences)

        train_loss /= len(x_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n = 0
        cor = 0.0
        with torch.no_grad():
            for sequences, scores in val_loader:
                n +=1
                sequences = torch.tensor(sequences).to(device)
                scores = torch.tensor(scores).to(device).unsqueeze(-1)
                outputs = model(sequences)
                correlation_coefficient, p_value = stats.spearmanr(outputs.squeeze().detach().cpu().numpy(), scores.detach().cpu().numpy())
                print("Spearman correlation coefficient:", correlation_coefficient)
                print("P-value:", p_value)
                loss = criterion(outputs.squeeze(), scores)
                val_loss += loss.item() * len(sequences)
                cor += correlation_coefficient

            val_loss /= len(x_val)
            cor /= n
 
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Cor: {cor:.4f}')
    
        train__loss.append(train_loss)
        test__loss.append(val_loss)
        correlation.append(cor)
    torch.save(model.state_dict(), "tnsfmr.pth")
    print("Saved PyTorch Model State to tnsfmr.pth")
    #Plot graphs
    plot_loss_vs_epochs(train__loss, test__loss)
    plot_cor_vs_epochs(correlation)
