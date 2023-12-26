import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split



# Define Attention layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Assuming x has shape (batch_size, seq_len, hidden_dim)
        energy = self.linear(x)
        weights = torch.softmax(energy, dim=1)
        output = torch.sum(weights * x, dim=1)
        return output