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


def evaluation(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for seq1, seq2, distance in data_loader:
            pred_distance = model(seq1, seq2, distance)
            loss = criterion(pred_distance, distance)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train(dataset_file, epoch_times):
    # Some hyperparameters
    split_rate = 0.8

    learning_rate = 0.001

    # toy examples
    # the_batch_size = 2
    # epoch_times = 5
    
    # real dataset
    the_batch_size = 64

    the_embedding_dim = 128
    the_hidden_dim = 128

    # toy example
    # sequences1, sequences2, distances = generate_toy_example()
    # And reflects the chars into numbers, by the operations in function load_dataset()

    # Load dataset
    print(f"Start loading dataset from {dataset_file}.\n")
    sequences1, sequences2, distances = load_dataset(dataset_file)
    print(f"Successfully loaded dataset with {len(sequences1)} sequences1, \
        {len(sequences2)} sequences2, \
        and {len(distances)} distances.\n")

    # Create dataset and DataLoader
    dataset = ProteinDataset(sequences1, sequences2, distances)
    train_size = int(split_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size])
    print(f"=************= Split datasets.\n")

    # Use pad_collate function for DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=the_batch_size, 
        shuffle=True, 
        collate_fn=pad_collate)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=the_batch_size, 
        shuffle=False, 
        collate_fn=pad_collate)
    print(f"=************= Load datasets, assigned pading function.\n")

    # Initialize the model
    model = ProteinDistanceModel(
        embedding_dim=the_embedding_dim, 
        hidden_dim=the_hidden_dim)
    print(f"=************= Initialize model.\n")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"=************= Define loss function and optimizer.\n")

    # Training loop
    print(f"=************= Start training.........\n")
    for epoch in range(epoch_times):
        model.train()
        for seq1, seq2, distance in train_loader:
            optimizer.zero_grad()
            pred_distance = model(seq1, seq2, distance)
            loss = criterion(pred_distance, distance)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        avg_loss = evaluation(model, test_loader, criterion)
        print(f"Epoch {epoch + 1}, Test Loss: {avg_loss:.4f}")

    print(f"\n=************= Finished training.\n")
    return model



def predict(dataset_file, model):
    print(f"=************= Start predicting...\n")
    ### Prediction
    # Set the model to evaluation mode (important if you have dropout layers)
    # model.eval()

    sequences1, sequences2, real_dis_tensor, selected_test_set = load_test_data(dataset_file)

    # Make predictions
    with torch.no_grad():
        # sequences1 = torch.stack(sequences1, dim = 0)
        # sequences2 = torch.tensor(sequences2, dtype=torch.long)
        predictions = model(sequences1, sequences2, real_dis_tensor)
        predict_distances = predictions.tolist()
        # print(f"Precited distances: {predict_distances}")

    print(f"protein1_pdb\tprotein2_pdb\tprotein1_classification\tprotein2_classfication\treal_distance\tpredict_distance\tdiff")
    for i in range(0, len(selected_test_set)):
        cur_record = selected_test_set[i]
        pro1 = cur_record["protein1_pdb"]
        pro2 = cur_record["protein2_pdb"]
        pro1_class = cur_record["protein1_classification"]
        pro2_class = cur_record["protein2_classfication"]
        real_distance = cur_record["distance"]
        predict_distance = predict_distances[i]
        diff = predict_distance[0] - real_distance
        print(f"{pro1}\t{pro2}\t{pro1_class}\t{pro2_class}\t{real_distance}\t{predict_distance}\t{diff}")

    print(f"\n=************= Finished predicting.\n")

