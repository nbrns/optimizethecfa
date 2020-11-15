"""Minimalistic DenseNet implementation, provided by Hannah Droege, University of Siegen."""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class DenseNet(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(DenseNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn

        kernel_size = 7
        kernel_size_small = 3
        padding = math.floor(kernel_size / 2)

        self.conv0 = nn.Conv2d(in_chn, 16, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(16 * 1 + in_chn, 16, kernel_size_small, padding=1)
        self.conv2 = nn.Conv2d(16 * 2 + in_chn, 16, kernel_size_small, padding=1)
        self.conv3 = nn.Conv2d(16 * 3 + in_chn, 16, kernel_size_small, padding=1)
        self.conv4 = nn.Conv2d(16 * 4 + in_chn, out_chn, kernel_size_small, padding=1)

    def forward(self, x):
        x = torch.cat((F.relu(self.conv0(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv1(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv2(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv3(x)), x), dim=1)
        x = self.conv4(x)
        return x
