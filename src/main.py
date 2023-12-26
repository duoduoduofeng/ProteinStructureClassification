import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from model.protein_distance_model import *



if __name__ == "__main__":
    dataset_file = "../generated_data/sample_proteins_dataset.txt"
    epoch_times = 5

    themodel = train(dataset_file, epoch_times)
    predict(dataset_file, themodel)