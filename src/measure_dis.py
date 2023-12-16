import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split



class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, sequences1, sequences2, distances):
        self.sequences1 = sequences1
        self.sequences2 = sequences2
        self.distances = distances

    def __len__(self):
        return len(self.sequences1)

    def __getitem__(self, idx):
        #print(f"idx = {idx}, seq1 = {self.sequences1[idx]}, \
        #    seq2 = {self.sequences2[idx]}, dist = {self.distances[idx]}\n")
        return self.sequences1[idx], \
            self.sequences2[idx], self.distances[idx]



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



# Define the model
class ProteinDistanceModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ProteinDistanceModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=21, 
            embedding_dim=embedding_dim)

        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, seq1, seq2, distance):
        embedded_seq1 = self.embedding(seq1)
        embedded_seq2 = self.embedding(seq2)

        attention_output_seq1 = self.attention(embedded_seq1)
        attention_output_seq2 = self.attention(embedded_seq2)

        # Concatenate attention outputs
        concatenated_output = torch.cat((
            attention_output_seq1, 
            attention_output_seq2), dim=1)

        # Predict distance
        pred_distance = self.fc(concatenated_output)
        return pred_distance




# Pad sequences
def pad_collate(batch):
    seq1, seq2, distances = zip(*batch)
    # print(f"New: Original data: seq1 = {seq1}, \
    #     \nseq2 = {seq2}, \ndistances = {distances}\n")

    pad_seq1 = pad_sequence(seq1, batch_first=True)
    pad_seq2 = pad_sequence(seq2, batch_first=True)
    transferred_dist = torch.stack(distances)
    
    # print(f"Padded data: (1)shape: \
    #     seq1: {pad_seq1.size()}, seq2: {pad_seq2.size()}\n\
    #     (2)data: seq1 = {pad_seq1}, \
    #     \nseq2 = {pad_seq2}, \ndistances = {transferred_dist}\n")
    return pad_seq1, pad_seq2, transferred_dist


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



def main():
    # Some hyperparameters
    split_rate = 0.8

    learning_rate = 0.001
    the_batch_size = 2
    epoch_times = 5

    the_embedding_dim = 64
    the_hidden_dim = 64

    # Dummy data
    # sequences1 = [torch.randint(0, 21, (20,)) for _ in range(20)]
    # sequences2 = [torch.randint(0, 21, (20,)) for _ in range(20)]
    # distances = torch.rand((20, 1))

    # Dummy data (replace with your actual data loading)
    sequences1 = [
        "TVDVGPDSVKSACIEVDIQQTFFDKTWPRPIDVSKADGIIYPQGRT", 
        "YSNITITYQGLFPYQGDHGDMYVYSAGHATGTTP", 
        "LFVANYSQDVKQFANGFVV",
        "MIVFVRFNSSHGFPVEVDSDTSIFQLKEVVAK",
        "GVPADQLRVIFAGKELRNDWTVQNCDL",
        "YSLYGVSGRGVFQNCTAVGVRQQRFVYDAYQNL",
        "MHVGYYPSNHIEVVSAYGLCDAANPTNCIAPVN",
        "LGNCVEYSLYGVSGRGVFQNCTAVGVRQQRFVYDAYQNLVGYYSDDGNYYCLRACVSVPVSVI",
        "FNLRNCTFMYTYNITEDEIL",
        "LIATVKKLTTPGKGLLA",
        "MSGLVPIVEPEVMIDGKHDIDTCQRVSEHV",
        "QVYNFKRLVFTNCNYN", 
        "AYTSSLLGSIAGVGWTAGLSSFAAIPF", 
        "VVEQAEGVECDFSPLLSGTPPQVYNFKRLVFT",
        "NCNYNLTKLLSLFSVNDFTCS",
        "QISPAAIASNCYSSLILDYFSYPLSMKSDLSVSSAGPISQFN",
        "YKQSFSNPTCLILATVPHNLTTITKPLKYSYINKC",
        "SRLLSDDRT",
        "EVPQLVNANQYSPCVSIVPSTVWED",
        "GDYYRKQLSPLEGGGWLVASGSTVAMTEQLQMGFGITVQ"
    ]
    sequences2 = [
        "QVYNFKRLVFTNCNYN", 
        "AYTSSLLGSIAGVGWTAGLSSFAAIPF", 
        "VVEQAEGVECDFSPLLSGTPPQVYNFKRLVFT",
        "NCNYNLTKLLSLFSVNDFTCS",
        "QISPAAIASNCYSSLILDYFSYPLSMKSDLSVSSAGPISQFN",
        "YKQSFSNPTCLILATVPHNLTTITKPLKYSYINKC",
        "SRLLSDDRT",
        "EVPQLVNANQYSPCVSIVPSTVWED",
        "GDYYRKQLSPLEGGGWLVASGSTVAMTEQLQMGFGITVQ",
        "ARMNSMAQLGKYKRSDDDASSSSLYV",
        "TVDVGPDSVKSACIEVDIQQTFFDKTWPRPIDVSKADGIIYPQGRT", 
        "YSNITITYQGLFPYQGDHGDMYVYSAGHATGTTP", 
        "LFVANYSQDVKQFANGFVV",
        "MIVFVRFNSSHGFPVEVDSDTSIFQLKEVVAK",
        "GVPADQLRVIFAGKELRNDWTVQNCDL",
        "YSLYGVSGRGVFQNCTAVGVRQQRFVYDAYQNL",
        "MHVGYYPSNHIEVVSAYGLCDAANPTNCIAPVN",
        "LGNCVEYSLYGVSGRGVFQNCTAVGVRQQRFVYDAYQNLVGYYSDDGNYYCLRACVSVPVSVI",
        "FNLRNCTFMYTYNITEDEIL",
        "LIATVKKLTTPGKGLLA"
    ]
    distances = 7 * torch.rand((21, 1))

    # Convert sequences to numerical indices
    char_to_index = {char: i for i, char in enumerate("ACDEFGHIKLMNPQRSTVWYX")}
    sequences1 = [torch.tensor([char_to_index[char] for char in seq]) \
                    for seq in sequences1]
    sequences2 = [torch.tensor([char_to_index[char] for char in seq]) \
                    for seq in sequences2]

    print(f"sequence1: {sequences1}\n")
    print(f"sequence2: {sequences2}\n")
    print(f"distances: {distances}\n")

    # Pad sequences
    # padded_sequences1 = pad_sequence(
    #     sequences1, 
    #     batch_first=True)
    # padded_sequences2 = pad_sequence(
    #     sequences2, 
    #     batch_first=True)
    # print(f"=************= Pad the sequences by library.\n")

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

    print(f"=************= Finished training.........\n")


if __name__ == "__main__":
    main()


