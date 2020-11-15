"""Module with custom loss function implementations."""
from typing import List
from torch import Tensor, einsum
import torch.nn
import torch
from network.input_type import get_loss_indices
from run.run_config import RunConfig


class MaskCrossEntropy:
    """Loss implementation to deal with masked labels"""
    def __init__(self, idc, run_config: RunConfig):
        """
        Initializes mask entropy loss.
        :param idc: List of classes to be valued in the loss
        :param run_config: Run config with further parameters
        """
        self.idc: List[int] = idc
        self.run_config: RunConfig = run_config
        self.log_softmax = torch.nn.LogSoftmax(dim=1)  # dim 1 dimension equals classes

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:

        log_softmax_p: Tensor = self.log_softmax(probs[:, self.idc, ...])
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        if self.run_config.balance_labels:
            v = 1 / mask.sum(3).sum(2)
            v = v / v.mean()
            for b in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    mask[b, c, :, :] = mask[b, c, :, :] * v[b, c]

        loss = - einsum("bcwh,bcwh->", mask, log_softmax_p)

        mask_sum = mask.sum()
        if mask_sum <= 0.1:
            raise ValueError('Mask is empty!')

        loss /= mask_sum
        return loss


class ColorVariationLoss:
    """Specific loss on color learning function. Punishes oscillating colors."""
    def __call__(self, layer):
        loss = (torch.abs(layer[:, 1:, :, :] - layer[:, :-1, :, :]) ** 2).sum()
        return loss


class TotalVariationLoss:
    """Total Variation Loss implementation. Punishes differences between adjacent output values"""
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, image: Tensor) -> Tensor:
        loss = self.total_variation_loss(image)
        return loss

    def total_variation_loss(self, image):
        image = self.softmax(image)
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
               torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss


class CombinedMaskEntropyTotalVariationLoss:
    """Combines Mask Entropy and Total Variation Loss"""
    def __init__(self, idc=[], alpha=0.1, run_config: RunConfig = None):
        self.idc = idc
        self.alpha = alpha
        self.mask_entropy = MaskCrossEntropy(idc=self.idc, run_config=run_config)
        self.total_variation = TotalVariationLoss()

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        mask_cross_entropy_loss = self.mask_entropy(probs, target)
        tv_loss = self.total_variation(probs)
        loss = mask_cross_entropy_loss + self.alpha * tv_loss
        return loss


class METVCVLoss:
    """Combines Mask Entropy, Total Variation and Color Variation Loss"""
    def __init__(self, run_config: RunConfig = None):
        """
        Initializes a METVCV Loss
        :param run_config: Run config with loss parameters
        """
        self.idc = run_config.idc if run_config.idc is not None else get_loss_indices(run_config.input_type)
        self.alpha = run_config.tvl_alpha
        self.beta = run_config.cvl_beta
        self.mask_entropy = MaskCrossEntropy(idc=self.idc, run_config=run_config)
        self.total_variation = TotalVariationLoss()
        self.color_variation = ColorVariationLoss()

    def __call__(self, probs: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        mask_cross_entropy_loss = self.mask_entropy(probs, target)
        tv_loss = self.total_variation(probs)
        cv_loss = self.color_variation(weights)
        loss = mask_cross_entropy_loss + self.alpha * tv_loss + self.beta * cv_loss
        return loss
