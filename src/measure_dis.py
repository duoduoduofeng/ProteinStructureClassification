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


def generate_toy_example():
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

    return sequences1, sequences2, distances



def load_dataset(dataset_file):
    sequences1 = []
    sequences2 = []
    distances = []

    print(f"The given dataset file is {dataset_file}.")

    err_input_data_count = 0
    row_num = 0
    with open(dataset_file, 'r') as fin:
        for line in fin:
            row_num += 1
            # tailor the head
            if row_num < 2:
                continue

            parts = line.strip().split('\t')
            if len(parts) == 7:
                distances.append(float(parts[2]))
                sequences1.append(parts[3])
                sequences2.append(parts[4])
            else:
                err_input_data_count += 1
        print(f"There are {err_input_data_count} error inputs \
            in the given dataset file {dataset_file}.")

    # Convert sequences to numerical indices
    abbrs = "ACDEFGHIKLMNPQRSTVWYX"
    char_to_index = {char: i for i, char in enumerate(abbrs)}
    print(f"char_to_index: {char_to_index}")
    sequences1 = [torch.tensor([char_to_index[char] for char in seq]) \
                    for seq in sequences1]
    sequences2 = [torch.tensor([char_to_index[char] for char in seq]) \
                    for seq in sequences2]

    distances_tensor = torch.tensor(distances, dtype=torch.float32).view(-1, 1)
    

    return sequences1, sequences2, distances_tensor



# This function is used to check the predict distance
def load_test_data(dataset_file):
    test_sets = []
    
    with open(dataset_file, 'r') as fin:
        row_num = 0
        field_names = []
        for line in fin:
            row_num += 1

            cur_pro = {}
            parts = line.strip().split('\t')
            if row_num == 1:
                field_names = parts
                # print(f"Fields names: {field_names}")
            else:
                for i in range(0, len(parts)):
                    cur_pro[field_names[i]] = parts[i]
                cur_pro["distance"] = float(cur_pro["distance"])
            test_sets.append(cur_pro)

    sequences1 = []
    sequences2 = []
    distances = []
    selected_test_set = []

    for i in range(0, 10):
        cur_record = test_sets[i * 100 + 1]
        selected_test_set.append(cur_record)

        distance = cur_record["distance"]
        seq1 = cur_record["protein1_seq"]
        seq2 = cur_record["protein2_seq"]

        sequences1.append(seq1)
        sequences2.append(seq2)
        distances.append(distance)

    # Convert sequences to numerical indices
    abbrs = "ACDEFGHIKLMNPQRSTVWYX"
    char_to_index = {char: i for i, char in enumerate(abbrs)}
    sequences1 = [torch.tensor([char_to_index[char] for char in seq]) \
                        for seq in sequences1]
    sequences2 = [torch.tensor([char_to_index[char] for char in seq]) \
                        for seq in sequences2]
    # Pad sequences, which is necessary
    sequences1 = pad_sequence(sequences1, batch_first=True)
    sequences2 = pad_sequence(sequences2, batch_first=True)

    real_dis_tensor = torch.tensor(distances, dtype=torch.float32).view(-1, 1)

    return sequences1, sequences2, real_dis_tensor, selected_test_set



def train(dataset_file):
    # Some hyperparameters
    split_rate = 0.8

    learning_rate = 0.001

    # toy examples
    # the_batch_size = 2
    # epoch_times = 5
    
    # real dataset
    the_batch_size = 64
    epoch_times = 600

    the_embedding_dim = 128
    the_hidden_dim = 128

    # toy example
    # sequences1, sequences2, distances = generate_toy_example()
    # And reflects the chars into numbers, by the operations in function load_dataset()

    # Load dataset
    print(f"Start loading dataset from {dataset_file}")
    sequences1, sequences2, distances = load_dataset(dataset_file)
    print(f"Successfully loaded dataset with {len(sequences1)} sequences1, \
        {len(sequences2)} sequences2, \
        and {len(distances)} distances")

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
    return model



def predict(dataset_file, model):
    print(f"=************= Start predicting...")
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

    print(f"Finished predicting.")



if __name__ == "__main__":
    dataset_file = "../generated_data/sample_proteins_dataset.txt"
    themodel = train(dataset_file)
    predict(dataset_file, themodel)


