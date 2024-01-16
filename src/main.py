from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from model.process import *



if __name__ == "__main__":
    common_path = "../generated_data/whole_pdbs/datasets/try_tp_3"
    
    ### 1. Training
    dataset_file = f"{common_path}/data/sample_proteins_dataset.train.txt"
    epoch_times = 100

    timestamp = time.time()
    datetime_object = datetime.fromtimestamp(timestamp)
    dt_object = datetime_object.strftime('%Y%m%d_%H%M%S')

    model_save_file = f"{common_path}/models/trained_model.dict.{dt_object}"
    train_log = f"{common_path}/logs/train.log.{dt_object}"
    print(f"model_save_file: {model_save_file}, \ntrain_log: {train_log}\n")

    train(dataset_file, model_save_file, train_log, epoch_times)

    ### 2. Prediction
    validate_dataset_file = "{common_path}/data/sample_proteins_dataset.validate.txt"

    #model_save_file = f"{common_path}/models/trained_model.dict.20240111_024648"

    predict_result_file = "{common_path}/result/predict_result.txt"
    predict(model_save_file, validate_dataset_file, predict_result_file)
