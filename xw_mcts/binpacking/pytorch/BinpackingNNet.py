import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# impala structure

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class BinPackingNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_h, self.board_w = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.in_channels = self.args.num_items + self.args.num_bins

        super(BinPackingNNet, self).__init__()
        # define the neural network structure for BinPacking game
        shape = (self.in_channels, self.board_h, self.board_w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=self.action_size)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        return F.log_softmax(logits, dim=1), torch.tanh(value)


# class BinPackingNNet(nn.Module):
#     def __init__(self, game, args):
#         # game params
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         self.args = args
#         self.in_channels = self.args.num_items + self.args.num_bins

#         super(BinPackingNNet, self).__init__()
#         # define the neural network structure for BinPacking game
#         # input channel, output channel, kernal size, stride
#         self.conv1 = nn.Conv2d(self.in_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
#         self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

#         self.bn1 = nn.BatchNorm2d(args.num_channels)
#         self.bn2 = nn.BatchNorm2d(args.num_channels)
#         self.bn3 = nn.BatchNorm2d(args.num_channels)
#         self.bn4 = nn.BatchNorm2d(args.num_channels)

#         self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
#         self.fc_bn1 = nn.BatchNorm1d(1024)

#         self.fc2 = nn.Linear(1024, 512)
#         self.fc_bn2 = nn.BatchNorm1d(512)

#         self.fc3 = nn.Linear(512, self.action_size)

#         self.fc4 = nn.Linear(512, 1)

#     def forward(self, s):
#         #                                                                s: batch_size x board_x x board_y * (num_bins+num_items)
#         s = s.view(-1, self.in_channels, self.board_x, self.board_y)  # batch_size x input_channels x board_x x board_y
#         s = F.relu(self.bn1(self.conv1(s)))                              # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn2(self.conv2(s)))                              # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn3(self.conv3(s)))                              # batch_size x num_channels x (board_x-2) x (board_y-2)
#         s = F.relu(self.bn4(self.conv4(s)))                              # batch_size x num_channels x (board_x-4) x (board_y-4)
#         s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

#         s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
#         s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

#         pi = self.fc3(s)                                                                         # batch_size x action_size
#         v = self.fc4(s)                                                                          # batch_size x 1

#         return F.log_softmax(pi, dim=1), torch.tanh(v)
