from pathlib import Path

import numpy as np
import h5py
import torch

from network.input_type import InputType
from run.run_config import RunConfig
from .preprocessing import LabelBuilder
from skimage.transform import resize
from PIL import Image


class HyperspectralImageProcessor:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    def load_takes(self):
        img = []
        for take in self.run_config.takes:
            img.append(self.load_take(take))

        # try batching them
        try:
            img = [torch.cat(img, 0)]
        except RuntimeError:
            print("Couldn't stack data tensors due to different sizes!")

        return img

    def load_take(self, take):
        if self.run_config.input_type == InputType.REAL_HYPERSPECTRAL or self.run_config.input_type == InputType.FRUITS_YELLOW:
            hyperspectral_file = Path(f'takes/{take}/hyperspectral.mat')
            arrays = {}
            try:
                f = h5py.File(hyperspectral_file)
            except OSError as err:
                print('Maybe you forgot to adjust takes?')
                raise err

            for k, v in f.items():
                arrays[k] = np.array(v)

            img = arrays['S']
            img = torch.from_numpy(img)
            img = img.rot90(3, (1, 2))

            resize_x = int(img.shape[1] * self.run_config.resize_factor / 2) * 2
            resize_y = int(img.shape[2] * self.run_config.resize_factor / 2) * 2
            resized_img = resize(img, (49, resize_x, resize_y))
            return torch.from_numpy(resized_img).unsqueeze(0)
        else:
            rgb_img_file = Path(f'takes/{take}/raw.png')
            img = torch.Tensor(np.array(Image.open(rgb_img_file))).permute((2, 0, 1))
            resize_x = int(img.shape[1] * self.run_config.resize_factor / 2) * 2
            resize_y = int(img.shape[2] * self.run_config.resize_factor / 2) * 2
            img = resize(img, (3, resize_x, resize_y))
            return torch.Tensor(img).unsqueeze(0)

    def load_labels_from_takes(self, takes):
        labels = []
        label_folder = 'labeling' if self.run_config.fully_supervised else 'masks'
        for take in takes:
            take_path = f'takes/{take}/{label_folder}/'
            labels.append(LabelBuilder(self.run_config).load_labels_from_folder(take_path))
        label_mapping = labels[0][1]
        label_data = [l[0] for l in labels]

        try:
            label_data = [torch.cat(label_data, dim=0)]
        except RuntimeError:
            print("Couldn't stack label tensors due to different sizes!")

        return label_data, label_mapping
