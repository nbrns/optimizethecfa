"""Module with custom layer implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil


class BayerLayer(nn.Module):
    """Layer that simulates a bayer pattern. Does a bayer pattern like selection of input channels, followed by
    an upscaling to original size"""

    def __init__(self, c_in, pattern_size, correct_shift, device):
        """
        Initializes bayer layer.
        :param c_in: input channels
        :param pattern_size: bayer pattern size
        :param correct_shift: correct pixel shift produced by bayer kernels
        :param device: execution device, e.g. cuda:0
        """
        super(BayerLayer, self).__init__()
        self.c_in = c_in
        self.pattern_size = pattern_size
        self.correct_shift = correct_shift
        self.kernels = self._generate_kernel().to(device)
        self.shift_kernels = self._generate_shift_kernels().to(device)
        self.kernels.requires_grad = False

    def _generate_kernel(self):
        """Generates the bayer pattern kernels for a layer"""
        weight_amount = self.pattern_size ** 2
        kernels = torch.zeros([weight_amount, self.c_in, weight_amount])

        for i in range(weight_amount):
            # kernel_nr, input_dim, weight_nr
            kernels[i][i % self.c_in][i] = 1
        kernels = kernels.view([weight_amount, self.c_in, self.pattern_size, self.pattern_size])
        return kernels

    def _generate_shift_kernels(self):
        weight_amount = self.pattern_size ** 2
        # c_in is unlike the normal bayer kernels always the same as pattern output
        c_in = weight_amount
        kernels = torch.zeros([weight_amount, c_in, weight_amount])

        for i in range(weight_amount):
            # kernel_nr, input_dim, weight_nr
            kernels[i][i % self.c_in][weight_amount-1-i] = 1
        kernels = kernels.view([weight_amount, c_in, self.pattern_size, self.pattern_size])
        return kernels

    def forward(self, x):
        upsample_shape = (x.shape[2], x.shape[3])
        x = F.conv2d(x, self.kernels, stride=2)
        x = F.interpolate(x, size=upsample_shape, mode='bicubic', align_corners=False)

        if self.correct_shift:
            if self.pattern_size not in [2, 3]:
                raise NotImplementedError('Can only correct shift on pattern size 2 or 3!')
            # for size 2
            padding = (1, 0, 1, 0)
            # for size 3
            if self.pattern_size == 3:
                padding = (1, 1, 1, 1)
            # correct pixel shift by applying "reverse order" kernels
            x = F.pad(x, padding, mode='replicate')
            x = F.conv2d(x, self.shift_kernels, stride=1, padding=0)

        return x


class CFASimulationLayer(nn.Module):
    """This layer simulates the a cfa, i.e. this is where color learning happens"""
    def __init__(self, c_in, c_out, device):
        """
        Initializes cfa simulation layer
        :param c_in: input channels
        :param c_out: output channels, i.e. learnable colors
        :param device: execution device, e.g. cuda:0
        """
        super(CFASimulationLayer, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0).to(device)
        self.device = device

    def clamp(self):
        self.conv.weight.data = self.conv.weight.data.clamp_(min=0).to(self.device)

    def forward(self, x):
        x = self.conv(x)
        return x
