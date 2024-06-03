import os
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch
import torchmetrics
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# import torch_geometric.loader.DataLoader

class MnistEEG_dataset(Dataset):
    def __init__(self, config, mode="train"):

        self.mode = mode
        self.num_channels = config["train_setting"]["num_channels"]
        self.num_samples = config["train_setting"]["num_samples"]
        self.num_classes = config["train_setting"]["num_classes"]
        self.num_trials = config["dataset"]["num_trials"]
        
        self.data = np.load('data/MnistEEG/processed_data_v3.npy', allow_pickle=True).item()
        self.x = torch.tensor(self.data['x']).to(torch.float32)
        self.y = torch.LongTensor(self.data['y']).flatten()
        
        data_size = self.x.shape[0]
        if mode == 'train':
            self.x = self.x[:int(data_size * 0.8)]
            self.y = self.y[:int(data_size * 0.8)]
        elif mode == 'val':
            self.x = self.x[int(data_size * 0.8):]
            self.y = self.y[int(data_size * 0.8):]
        
        print(self.x.shape, self.y.shape)
        
        if config["dataset"]["preprocess"]:
            print("Preprocessing...")
            self.__preprocess__()
        else:
            print("No preprocess")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = {
            'x': self.x[idx],
            'y': self.y[idx],
        }
        return sample
    
    def __preprocess__(self):
        
        x, y = [], []
        for class_id in range(self.num_classes):
            single_trial_x = self.x[self.y.reshape(-1) == class_id]
            
            # take mean of adj trials
            multi_trial_x = []
            for i in range(0, single_trial_x.shape[0] - self.num_trials + 1):
                multi_trial_x.append(single_trial_x[i:i+self.num_trials].mean(dim=0))
        
            x.append(torch.stack(multi_trial_x))
            y.append(torch.tensor([class_id] * len(multi_trial_x)))

        self.x = torch.cat(x)
        self.y = torch.cat(y)
        
        # shuffle
        indices = np.arange(self.x.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)
        self.x, self.y = self.x[indices], self.y[indices].reshape(-1,1)
        
        return

class LargeEEG_dataset(Dataset):
    def __init__(self, config, mode="train"):

        self.mode = mode
        self.num_channels = config["train_setting"]["num_channels"]
        self.num_samples = config["train_setting"]["num_samples"]
        self.num_classes = config["train_setting"]["num_classes"]
        self.dataset_id = config["dataset"]["dataset_id"]
        self.num_trials = config["dataset"]["num_trials"]
        
        self.dataset_id = self.dataset_id[:-1]+'0'
        self.data = np.load(f'data/LargeEEG/processed/{self.dataset_id}.npy', allow_pickle=True).item()        
        if mode == 'train':
            self.x = self.data['train_x']
            self.y = self.data['train_y']
        elif mode == 'val':
            self.x = self.data['val_x']
            self.y = self.data['val_y']
        elif mode == 'test':
            self.x = self.data['test_x']
            self.y = self.data['test_y']
        self.x = torch.tensor(self.x).to(torch.float32)
        self.y = torch.LongTensor(self.y).flatten()
        
        print(self.x.shape, self.y.shape)
        
        if config["dataset"]["preprocess"]:
            print("Preprocessing...")
            self.__preprocess__()
        else:
            print("No preprocess")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = {
            'x': self.x[idx],
            'y': self.y[idx],
        }
        return sample
    
    def __preprocess__(self):
        
        x, y = [], []
        for class_id in range(self.num_classes):
            single_trial_x = self.x[self.y.reshape(-1) == class_id]
            
            # take mean of adj trials
            multi_trial_x = []
            for i in range(0, single_trial_x.shape[0] - self.num_trials + 1):
                multi_trial_x.append(single_trial_x[i:i+self.num_trials].mean(dim=0))
        
            x.append(torch.stack(multi_trial_x))
            y.append(torch.tensor([class_id] * len(multi_trial_x)))

        self.x = torch.cat(x)
        self.y = torch.cat(y)
        
        # shuffle
        indices = np.arange(self.x.shape[0])
        np.random.seed(0)
        np.random.shuffle(indices)
        self.x, self.y = self.x[indices], self.y[indices].reshape(-1,1)
        
        return
# --------------------------------------------------------------------------------

class normal_loader():
    def __init__(self, config, mode):
        
        self.batch_size = config["train_setting"]["batch_size"]
        self.num_workers = config["train_setting"]["num_workers"]
        self.dataset_name = config["dataset"]["name"]
        
        if mode == 'train':
            self.train_loader = self._generate_loader(config, mode="train")
            self.val_loader = self._generate_loader(config, mode="val")
        elif mode == 'test':
            self.test_loader = self._generate_loader(config, mode="test")
        
    def _generate_loader(self, config, mode):
        if self.dataset_name == 'MnistEEG':
            dataset = MnistEEG_dataset(config, mode)
        elif self.dataset_name == 'LargeEEG':
            dataset = LargeEEG_dataset(config, mode)
            
        shuffle = True if mode == 'train' else False
        loader = DataLoader(dataset, batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            shuffle=shuffle,
                            pin_memory=True,
                            )
        return loader


if __name__ == "__main__":
    config = {
        "train_setting": {
            "batch_size": 2,
            "num_workers": 4,
            "num_channels": 17,
            "num_samples": 100,
            "num_classes": 17,
            "sub_id": 1,
            
        },
        "dataset": {
            "root_path": "C:/Users/112062522/Downloads/112062522_whuang/research/GNN/project2_2/data/processed_v2/",
            # "root_path": "C:/Users/112062522/Downloads/112062522_whuang/research/GNN/project/data/processed/",
            "num_trials": 1,
        }
    }

    
    
    
    normal_loader = v2_normal(config)
    for i, data in enumerate(normal_loader.train_loader):
        print(data['x'].shape, data['y'].shape)
        # print(i)
        break
    # for data in normal_loader.val_loader:
    #     print(data['x'].shape, data['y'].shape)
    #     break