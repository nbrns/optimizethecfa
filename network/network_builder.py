"""Module to enumerate networks. It has the intention to simplify access to network architectures and also, fixing
the architectures."""
from enum import Enum
from network.networks import BayerDenseNet, BayerUNet, BayerPatternNet, UNet, DenseNet, ReducedUNet
from run.run_config import RunConfig


class NetworkIdentifier(Enum):
    BAYER_UNET = 1
    BAYER_DENSENET = 2
    CONV_40_1 = 3
    CONV_40_3 = 4
    CONV_40_5 = 5
    CONV_40_7 = 6
    CONV_40_9 = 7
    UNET = 8
    DENSE_NET = 9
    REDUCED_UNET = 10


def get_network(run_config: RunConfig):
    network = run_config.network_identifier
    model = None
    device = run_config.device
    if network == NetworkIdentifier.BAYER_UNET:
        model = BayerUNet(run_config).to(device)
    if network == NetworkIdentifier.BAYER_DENSENET:
        model = BayerDenseNet(run_config).to(device)
    if network == NetworkIdentifier.CONV_40_1:
        model = BayerPatternNet(40, 40, 40, kernel_size=1, padding=0, run_config=run_config).to(device)
    if network == NetworkIdentifier.CONV_40_3:
        model = BayerPatternNet(40, 40, 40, kernel_size=3, padding=1, run_config=run_config).to(device)
    if network == NetworkIdentifier.CONV_40_5:
        model = BayerPatternNet(40, 40, 40, kernel_size=5, padding=2, run_config=run_config).to(device)
    if network == NetworkIdentifier.CONV_40_7:
        model = BayerPatternNet(40, 40, 40, kernel_size=7, padding=3, run_config=run_config).to(device)
    if network == NetworkIdentifier.CONV_40_9:
        model = BayerPatternNet(40, 40, 40, kernel_size=9, padding=4, run_config=run_config).to(device)
    if network == NetworkIdentifier.UNET:
        model = UNet(run_config=run_config).to(device)
    if network == NetworkIdentifier.DENSE_NET:
        model = DenseNet(run_config=run_config).to(device)
    if network == NetworkIdentifier.REDUCED_UNET:
        model = ReducedUNet(run_config=run_config).to(device)

    if model is None:
        raise TypeError('No network matched!')

    return model
