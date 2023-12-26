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
    print(f"char_to_index: {char_to_index}.\n")
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
        cur_record = test_sets[i * 100 + 10]
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
