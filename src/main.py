from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from model.process import *



if __name__ == "__main__":
    common_path = "../generated_data/whole_pdbs/datasets/try_cl_1000001"
    
    ### 1. Training
    dataset_file = f"{common_path}/data/sample_proteins_dataset.train.txt"
    epoch_times = 150

    timestamp = time.time()
    datetime_object = datetime.fromtimestamp(timestamp)
    dt_object = datetime_object.strftime('%Y%m%d_%H%M%S')

    the_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", the_device)

    model_save_file = f"{common_path}/models/trained_model.dict.{dt_object}"
    train_log = f"{common_path}/logs/train.log.{dt_object}"
    print(f"model_save_file: {model_save_file}, \ntrain_log: {train_log}\n")
    
    the_batch_size = 1024
    train(dataset_file, model_save_file, train_log, the_device, epoch_times, the_batch_size)

    ### 2. Prediction
    validate_dataset_file = f"{common_path}/data/sample_proteins_dataset.validate.txt"
    predict_result_file = f"{common_path}/result/predict_result.validate.txt.{dt_object}"
    predict(model_save_file, validate_dataset_file, predict_result_file, the_device)

    validate_dataset_file = f"{common_path}/data/sample_proteins_dataset.excluded_validate.txt"
    predict_result_file = f"{common_path}/result/predict_result.excluded_validate.txt.{dt_object}"
    predict(model_save_file, validate_dataset_file, predict_result_file, the_device)
