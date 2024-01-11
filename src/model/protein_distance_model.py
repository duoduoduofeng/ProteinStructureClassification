import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from preprocess.protein_dataset import *
from model.attention_layer import *


# Define the model
class ProteinDistanceModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ProteinDistanceModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=21, 
            embedding_dim=embedding_dim)

        self.attention = Attention(hidden_dim)

        self.fc1 = nn.Linear(2 * hidden_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 1)

    def forward(self, seq1, seq2, distance):
        embedded_seq1 = self.embedding(seq1)
        embedded_seq2 = self.embedding(seq2)

        attention_output_seq1 = self.attention(embedded_seq1)
        attention_output_seq2 = self.attention(embedded_seq2)

        # Concatenate attention outputs
        concatenated_output = torch.cat((
            attention_output_seq1, 
            attention_output_seq2), dim=1)

        x = self.fc1(concatenated_output)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Predict distance
        
        pred_distance = self.fc3(x)
        # The logstic regression doesn't help here. (distance between 0 and 1)
        # pred_distance = torch.sigmoid(self.fc3(x))
        return pred_distance
