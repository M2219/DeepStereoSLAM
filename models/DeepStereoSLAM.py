import argparse
import json

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from typing import List, Optional, Tuple
from .core.raft import RAFT
from .submodule import *
torch.autograd.set_detect_anomaly(True)
def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(parser):
    entry = parser.parse_args()
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default="models/cf.json", help='experiment configure file name', type=str)
parser.add_argument('--path', help='checkpoint path', type=str, default=None)
parser.add_argument('--url', default="MemorySlices/Tartan-C-T-TSKH-kitti432x960-M", help='checkpoint url', type=str)
parser.add_argument('--device', help='inference device', type=str, default='cpu')
args = parse_args(parser)

class SubModule(nn.Module):
    def __init__(self) -> None:
        super(SubModule, self).__init__()

    def weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self) -> None:
        super(Feature, self).__init__()
        model = RAFT.from_pretrained(args.url, args=args)

        self.conv1 = model.cnet.conv1
        self.bn1 = model.cnet.bn1
        self.relu = model.cnet.relu

        self.l1 = torch.nn.Sequential(*model.cnet.layer1)
        self.l2 = torch.nn.Sequential(*model.cnet.layer2)
        self.l3 = torch.nn.Sequential(*model.cnet.layer3)
        self.final_conv = model.cnet.final_conv


        #self.conv4 = nn.Sequential(BasicConv(256, 512, is_3d=False, bn=True, gelu=True, kernel_size=3,
        #                                     padding=1, stride=2, dilation=1),
        #                           BasicConv(512, 512, is_3d=False, bn=True, gelu=True, kernel_size=3,
        #                                     padding=1, stride=1, dilation=1))


        #self.final_conv = BasicConv(512, 512, is_3d=False, bn=True, gelu=True, kernel_size=1,
        #                                     padding=1, stride=1, dilation=1)

        #self.conv5 = nn.Sequential(BasicConv(512, 1024, is_3d=False, bn=True, gelu=True, kernel_size=3,
        #                                     padding=1, stride=2, dilation=1),
        #                           BasicConv(1024, 1024, is_3d=False, bn=True, gelu=True, kernel_size=3,
        #                                     padding=1, stride=1, dilation=1))

        #self.final_conv = BasicConv(1024, 1024, is_3d=False, bn=True, gelu=True, kernel_size=1,
        #                                     padding=1, stride=1, dilation=1)

    def forward(self, x: torch.Tensor) -> List[ torch.Tensor]:
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.l1)):
            x = self.l1[i](x)
        for i in range(len(self.l2)):
            x = self.l2[i](x)
        for i in range(len(self.l3)):
            x = self.l3[i](x)

        #x = self.conv4(x)
        #x = self.conv5(x)
        x = self.final_conv(x)

        return x

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

        self.conv_gru1 = ConvGRUCell(256, 128, kernel_size=3)
        self.gru_head = GruHead(128, 256, 256)
        self.conv_gru2 = ConvGRUCell(256, 128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=938496, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=6)

        self.h_t1 = None
        self.h_t2 = None

        self.relu = nn.ReLU()

        #def forward(self, img: torch.Tensor, self.h_t1, h_t2) -> List[torch.Tensor]:
    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:

        img = 2 * (img / 255.0) - 1.0
        img = torch.cat((img[:, 1:], img[:, :-1]), dim=2).squeeze(1)

        batch_size = img.size(0)

        print("input", img.shape)

        features = self.feat_out(img)

        print("features,", features.shape)
        if self.h_t1 is not None:
            print("self.h_t1_1", self.h_t1.shape)

        self.h_t1 = self.conv_gru1(features, self.h_t1)
        print("self.h_t1_2", self.h_t1.shape)

        print("self.h_t1_3", self.h_t1.shape)
        h_h = self.gru_head(self.h_t1)
        print("self.h_t1_4", h_h.shape)

        self.h_t2 = self.conv_gru2(h_h, self.h_t2)

        nn_inp = self.h_t2.view(batch_size, -1)
        print("output",  nn_inp.shape)

        nn_inp = self.relu(self.fc1(nn_inp))
        poses = self.fc2(nn_inp)

        print("poses",  poses.shape)

        return poses  #self.h_t1, h_t2
