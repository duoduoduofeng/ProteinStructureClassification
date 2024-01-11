from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from model.process import *



if __name__ == "__main__":
    dataset_file = "../generated_data/train/sample_proteins_dataset.txt"
    validate_dataset_file = "../generated_data/validate/sample_proteins_dataset.validate.txt"

    timestamp = time.time()
    datetime_object = datetime.fromtimestamp(timestamp)
    dt_object = datetime_object.strftime('%Y%m%d_%H%M%S')
    model_save_file = f"../generated_data/train/models/trained_model.dict.{dt_object}"
    train_log = f"../generated_data/train/logs/train.log.{dt_object}"
    print(f"model_save_file: {model_save_file}, \ntrain_log: {train_log}\n")

    predict_result_file = "../generated_data/predict_result.txt"

    epoch_times = 1000

    # themodel = train(dataset_file, epoch_times)
    # predict(themodel, validate_dataset_file, predict_result_file)
    train(dataset_file, model_save_file, train_log, epoch_times)
    predict(model_save_file, validate_dataset_file, predict_result_file)
