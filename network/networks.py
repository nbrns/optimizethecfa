"""Module containing network implementations. Strongly cooperates with the network_builder module."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.custom_layers import BayerLayer, CFASimulationLayer
from network.densenet import DenseNet as OriginalDenseNet
from network.unet import UNet as OriginalUNet
from network.unet import WeakUNet
from network.input_type import get_dimensions, get_class_number
from run.run_config import RunConfig

# Uses bayer pattern
class BayerPatternNet(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, kernel_size=1, padding=0, run_config: RunConfig = None):
        super(BayerPatternNet, self).__init__()
        self.run_config = run_config
        d_in = get_dimensions(run_config.input_type)
        d_out = get_class_number(run_config.input_type)
        self.cfa_sim_layer = CFASimulationLayer(d_in, run_config.cfa_sim_out_channels, run_config.device)
        self.bayer_layer = BayerLayer(run_config.cfa_sim_out_channels,
                                      run_config.bayer_pattern_size,
                                      run_config.correct_pixel_shift,
                                      run_config.device
                                      )

        self.conv1 = nn.Conv2d(run_config.bayer_pattern_size ** 2, hidden1, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(hidden2, hidden3, kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(hidden3, d_out, kernel_size, padding=padding)

    def forward(self, x):
        x = self.cfa_sim_layer(x)
        x = self.bayer_layer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class BayerUNet(nn.Module):
    def __init__(self, run_config: RunConfig):
        super(BayerUNet, self).__init__()
        self.run_config = run_config
        d_in = get_dimensions(run_config.input_type)
        self.cfa_sim_layer = CFASimulationLayer(d_in, run_config.cfa_sim_out_channels, run_config.device).to(
            run_config.device)
        self.bayer_layer = BayerLayer(run_config.cfa_sim_out_channels, run_config.bayer_pattern_size,
                                      run_config.correct_pixel_shift, run_config.device)
        self.u_net = UNet(self.run_config)

    def forward(self, x):
        x = self.cfa_sim_layer(x)
        x = self.bayer_layer(x)
        x = self.u_net(x)
        return x


class BayerDenseNet(nn.Module):
    def __init__(self, run_config: RunConfig):
        super(BayerDenseNet, self).__init__()
        self.run_config = run_config
        d_in = get_dimensions(run_config.input_type)
        self.cfa_sim_layer = CFASimulationLayer(d_in, run_config.cfa_sim_out_channels, run_config.device).to(
            run_config.device)
        self.bayer_layer = BayerLayer(run_config.cfa_sim_out_channels, run_config.bayer_pattern_size,
                                      run_config.correct_pixel_shift, run_config.device)
        self.dense_net = DenseNet(run_config)

    def forward(self, x):
        x = self.cfa_sim_layer(x)
        x = self.bayer_layer(x)
        x = self.dense_net(x)
        return x


class UNet(nn.Module):
    def __init__(self, run_config: RunConfig):
        super(UNet, self).__init__()
        in_channels = run_config.bayer_pattern_size ** 2
        n_classes = get_class_number(run_config.input_type)
        self.u_net = OriginalUNet(in_channels, n_classes, width=4)

    def forward(self, x):
        x = self.u_net(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, run_config: RunConfig):
        super(DenseNet, self).__init__()
        in_channels = run_config.bayer_pattern_size ** 2
        n_classes = get_class_number(run_config.input_type)
        self.dense_net = OriginalDenseNet(in_channels, n_classes)

    def forward(self, x):
        x = self.dense_net(x)
        return x


class ReducedUNet(nn.Module):
    def __init__(self, run_config: RunConfig):
        super(ReducedUNet, self).__init__()
        in_channels = run_config.bayer_pattern_size ** 2
        n_classes = get_class_number(run_config.input_type)
        c_in = get_dimensions(run_config.input_type)
        self.cfa_sim_layer = CFASimulationLayer(c_in, run_config.cfa_sim_out_channels, run_config.device)
        self.bayer_layer = BayerLayer(run_config.cfa_sim_out_channels, run_config.bayer_pattern_size,
                                      run_config.correct_pixel_shift, run_config.device)
        self.u_net = WeakUNet(run_config.bayer_pattern_size**2, n_classes, width=4)

    def forward(self, x):
        x = self.cfa_sim_layer(x)
        x = self.bayer_layer(x)
        x = self.u_net(x)
        return x
