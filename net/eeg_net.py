import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class EEGNet_Model(nn.Module):
    def __init__(self, config):
        super(EEGNet_Model, self).__init__()
        
        self.kernel_length1 = config["backbone"]["kernel_length1"]
        self.kernel_length2 = config["backbone"]["kernel_length2"]
        self.pooling_size1 = config["backbone"]["pooling_size1"]
        self.pooling_size2 = config["backbone"]["pooling_size2"]
        
        self.dropout = config["backbone"]["dropout"]
        self.D = config["backbone"]["D"]
        self.F1 = config["backbone"]["F1"]
        # self.F2 = config["backbone"]["F2"]
        self.F2 = self.F1 * self.D
        self.num_channels = config["train_setting"]["num_channels"]
        self.num_classes = config["train_setting"]["num_classes"]
        self.num_samples = config["train_setting"]["num_samples"]

        # Block 1
        self.conv2d_block1 = nn.Conv2d(1, self.F1, (1, self.kernel_length1), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthwise_conv2d = nn.Conv2d(self.F1, self.F1 * self.D, kernel_size=(self.num_channels, 1), groups=self.F1)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, self.pooling_size1))
        self.dropout1 = nn.Dropout(p=self.dropout)
        
        # Block 2
        self.separable_conv2d = nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, self.kernel_length2), groups=self.F1 * self.D, padding='same')
        self.pointwise_conv2d = nn.Conv2d(self.F2, self.F2, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm2d(self.F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, self.pooling_size2))
        self.dropout2 = nn.Dropout(p=self.dropout)
        
        # Dense
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.F2 * int(self.num_samples / self.pooling_size1 / self.pooling_size2), self.num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.F2 * int(self.num_samples / self.pooling_size1 / self.pooling_size2), config["backbone"]["tmp"]),
        #     nn.BatchNorm1d(config["backbone"]["tmp"]),
        #     nn.ReLU(),
        #     nn.Linear(config["backbone"]["tmp"], self.num_classes),
        # )
    
    def forward(self, x):
        
        # x: (batch_size, 1, n_channel, n_samples)
        B, C, H, W = x.shape
        
        # Block 1
        x = self.conv2d_block1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv2d(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_conv2d(x)
        x = self.pointwise_conv2d(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Flatten and dense
        x = self.flatten(x)
        x = self.dense(x)
        # x = x.reshape(B, -1)
        # x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    
    # 128 hz, 2s
    # 64 hz,  0.5s

    config = {
        "backbone": {
            "kernel_length1": 50,    # 0.5s for 64 hz: 32
            "kernel_length2": 25,   # 1/4s for 32 hz: 8
            "pooling_size1": 7,    # downsample 64 hz to 32 hz
            "pooling_size2": 7,    # 8
            "dropout": 0.5,
            "D": 2,
            "F1": 8,
            "F2": 16,
        },
        "train_setting": {
            "num_channels": 17,
            "num_classes": 1654,
            "num_samples": 100,
        }
    }    
    net = EEGNet_Model(config)
    print(net)
    print(net.forward(Variable(torch.Tensor(np.random.rand(3, 1, 17, 100)))))
    print(net.forward(Variable(torch.Tensor(np.random.rand(3, 1, 17, 100)))).shape)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())