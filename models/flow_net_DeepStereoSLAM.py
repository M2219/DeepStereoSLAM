import argparse
import json

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from torch.nn.init import kaiming_normal_, orthogonal_

from typing import List, Optional, Tuple
from .core.raft import RAFT
from .submodule import *
from .convgru import ConvGRU

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)#, inplace=True)
        )

class Feature(nn.Module):
    def __init__(self) -> None:
        super(Feature, self).__init__()
        self.batchNorm = True
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4   = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv5   = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv5_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv6   = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.5)

        #self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=0.2)
        #self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=0.2)
        #self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=0.2)
        #self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        #self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=0.2)
        #self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=0.2)
        #self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=0.2)
        #self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=0.2)
        #self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> List[ torch.Tensor]:

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)

        return out_conv5

class GruHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256):
        super(GruHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DeepStereoSLAM(nn.Module):
    def __init__(self):
        super(DeepStereoSLAM, self).__init__()
        self.feat_out = Feature()

        self.conv_gru = ConvGRU(input_size=256, hidden_sizes=[128, 128], kernel_sizes=[3, 3], n_layers=2)

        #self.conv_gru1 = ConvGRUCell(256, 128, kernel_size=3)
        #self.gru_head = GruHead(128, 256, 256)
        #self.conv_gru2 = ConvGRUCell(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=938496, out_features=6)
        #self.fc2 = nn.Linear(in_features=256, out_features=6)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.h_tt1 = None
        self.h_tt2 = None

        #self.dropout1 = nn.Dropout(p=0.3)  # Dropout after GRU1
        #self.dropout2 = nn.Dropout(p=0.3)  # Dropout after GRU2

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    #def forward(self, img: torch.Tensor, h_tt1, h_tt2) -> List[torch.Tensor]:
    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:

        #img = 2 * (img / 255.0) - 1.0
        img = torch.cat((img[:, 1:], img[:, :-1]), dim=2).squeeze(1)
        batch_size = img.size(0)

        #print("input", img.shape)

        features = self.feat_out(img)

        print("features,", features.shape)
        print("feat output", features.mean())

        #h_t1 = self.conv_gru1(features, self.h_tt1 if self.h_tt1 is not None else None)
        #h_t1 = self.batch_norm1(h_t1)
        #h_t1 = self.dropout1(h_t1)  # Apply Dropout after GRU2

        #print("h_t1_2", h_t1.shape)

        #print("h_t1_3", h_t1.shape)
        #in_gru2 = self.gru_head(h_t1)

        #print("h_t1_4", h_h.shape)
        #h_t2 = self.conv_gru2(h_t1, self.h_tt2 if self.h_tt2 is not None else None)

        #h_t2 = self.dropout2(h_t2)  # Apply Dropout after GRU2
        #print("gru output", h_t1.mean(), h_t2.mean())
        #if self.h_tt1 is not None:
        #    print("gru output", self.h_tt1.mean(), self.h_tt2.mean())

        #self.h_tt1 = h_t1.detach()
        #self.h_tt2 = h_t2.detach()
        h_t2 = self.conv_gru(features)
        #h_t = self.batch_norm2(h_t2[-1])
        #print(h_t.mean())
        nn_inp = h_t2[-1].view(batch_size, -1)

        poses = self.relu(self.fc1(nn_inp))
        #h_t1 = self.dropout1(poses)  # Apply Dropout after GRU1

        #poses = self.fc2(nn_inp)

        #print("poses",  poses.shape)

        return poses #, h_t1, h_t2
