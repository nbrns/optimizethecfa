"""Module for data preprocessing, e.g. building samples and labels from given takes / data."""
import os
import cv2
import numpy as np
import rawpy
import torch
from pathlib import Path
from PIL import Image

from network.input_type import InputType
from run.run_config import RunConfig


class RawImageProcessor:

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    def load_takes(self):
        img = []
        for take_name in self.run_config.takes:
            channels_path = Path(f'takes/{take_name}/channels/')
            try:
                img.append(self.preprocess_folder(data_folder=channels_path, verbose=self.run_config.verbose))
            except FileNotFoundError as err:
                print('Maybe you are haven\'t set hyperspectral parameter while training on hyperspectral data?')
                raise err

        img = [torch.cat(img, dim=0)]
        return img

    def preprocess_folder(self, data_folder=Path('./'), verbose=False):
        data_folder = Path(data_folder)
        images = []
        if self.run_config.input_type == InputType.SIM_HYPER_ONLY_RGB:
            images = [self._load_and_process(file, verbose) for file in sorted(data_folder.iterdir()) if
                      file.name == 'flash.CR2']
        if self.run_config.input_type == InputType.SIM_HYPERSPECTRAL:
            images = [self._load_and_process(file, verbose) for file in sorted(data_folder.iterdir()) if
                  file.suffix == '.CR2']
        stacked_images = np.concatenate(images, axis=2)
        permuted_images = torch.from_numpy(stacked_images).permute(2, 0, 1).unsqueeze(0)
        return permuted_images

    def _load_and_process(self, file, verbose=False):
        if verbose:
            print(f'Processing: {file}')
        raw_color = rawpy.ColorSpace(0)
        absolute_path = str(file)
        raw = rawpy.imread(absolute_path)

        img = raw.postprocess(half_size=True,
                              output_color=raw_color,
                              output_bps=16,
                              gamma=(1, 1),
                              no_auto_bright=True
                              ).astype(float) / 65535
        if self.run_config.resize_factor != 0:
            img = cv2.resize(img, dsize=(int(img.shape[1] * self.run_config.resize_factor / 2) * 2,
                                         int(img.shape[0] * self.run_config.resize_factor / 2) * 2),
                             interpolation=cv2.INTER_CUBIC)
        return img


class LabelBuilder:

    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    def load_labels_from_folder(self, label_folder=Path('labeling/')):
        mask_items = os.listdir(label_folder)
        labels = []

        # build labels
        label_nr = 0
        label_mapping = {}
        # sorted is required to work on different computers
        for file in sorted(mask_items):
            if self.run_config.verbose:
                print(f'Processing: {file}')
            image = np.asarray(Image.open(os.path.join(label_folder, file)))
            if self.run_config.resize_factor != 0:
                resize_x = int(image.shape[1] * self.run_config.resize_factor / 2) * 2
                resize_y = int(image.shape[0] * self.run_config.resize_factor / 2) * 2
                image = cv2.resize(image, dsize=(resize_x, resize_y), interpolation=cv2.INTER_NEAREST)
            else:
                resize_x = int(image.shape[1])
                resize_y = int(image.shape[0])
            mask = np.sum(image, axis=2)
            mask = (mask == np.max(mask)).astype(int)

            label_values = mask[..., np.newaxis]
            labels.append(label_values)
            label_name = str(file).split('.')[1]
            label_mapping.update({label_nr: label_name})
            label_nr += 1

        labels = np.concatenate(labels, axis=2)

        # add throwaway class for unlabeled pixels
        unlabeled = np.zeros((resize_y, resize_x, 1))
        unlabeled_indices = np.where(labels.sum(axis=2) == 0)
        unlabeled[unlabeled_indices] = 1
        labels = np.append(labels, unlabeled, axis=2)
        permuted_labels = torch.from_numpy(labels).permute(2, 0, 1).unsqueeze(0)
        return permuted_labels, label_mapping

    def load_labels_from_take(self, takes=[1], verbose=False):
        labels = []
        label_folder = 'labeling' if self.run_config.fully_supervised else 'masks'
        for take_name in takes:
            take_path = Path(f'takes/{take_name}/{label_folder}/')
            if verbose:
                print(f'Processing take: {take_path}')
            labels.append(self.load_labels_from_folder(take_path))
        label_mapping = labels[0][1]
        label_data = [l[0] for l in labels]
        return [torch.cat(label_data, dim=0)], label_mapping





