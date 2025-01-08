import argparse
import os
import time
import yaml

import torch
import numpy as np
import pandas as pd
import pickle

from torch.utils.data import DataLoader

from dataset import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset

parser = argparse.ArgumentParser(description="DeepStereoSLAM")
parser.add_argument("--pathToSettings", default="configs.yaml", help="path to settings")

args = parser.parse_args()

with open(args.pathToSettings, 'r') as f:
    fsSettings = yaml.load(f, Loader=yaml.SafeLoader)

if not os.path.isfile("train_data_info_path.pkl"):
    print("generate data info")
    train_df = get_data_info("train", fsSettings, overlap=1, sample_times=1)
    valid_df = get_data_info("validation", fsSettings, overlap=1, sample_times=1)

    train_df.to_pickle("train_data_info_path.pkl")
    valid_df.to_pickle("valid_data_info_path.pkl")

with open('train_data_info_path.pkl', 'rb') as file:
    train_df = pickle.load(file)

with open('valid_data_info_path.pkl', 'rb') as file:
    valid_df = pickle.load(file)

train_sampler = SortedRandomBatchSampler(train_df, fsSettings["batch_size"], drop_last=True)
train_dataset = ImageSequenceDataset(train_df)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)

valid_sampler = SortedRandomBatchSampler(valid_df, fsSettings["batch_size"], drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=4)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('='*50)



