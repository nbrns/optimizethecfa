import torch
import numpy as np
from scipy.io import loadmat
from network.input_type import InputType, is_urban_input, is_only_rgb, get_class_number, is_urban_only_rgb_input
from preprocessing.preprocessing import LabelBuilder
from run.run_config import RunConfig
from PIL import Image

class UrbanImageProcessor:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.input_type = run_config.input_type

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
        data = loadmat(f'takes/{take}/hyperspectral.mat')

        if is_urban_input(self.run_config.input_type):
            hyperspectral_data = data['Y'].reshape((162, 307, 307))
            return torch.from_numpy(hyperspectral_data.astype('int16')).unsqueeze(0)
        if is_urban_only_rgb_input(self.input_type):
            # band selection for rgb
            # r, g, b = (13, 9, 0)
            # rgb_data = data['Y'][[r, g, b]].reshape((3, 307, 307))
            img = torch.Tensor(np.array(Image.open(f'takes/{take}/rgb.png'))).permute((2, 0, 1))
            return img.unsqueeze(0)

        raise TypeError('No valid Input Type specified!')


class UrbanLabelProcessor:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config

    def load_labels(self):
        labels = []
        label_mapping = {
            0: 'road',
            1: 'grass',
            2: 'tree',
            3: 'roof'
        }
        if len(self.run_config.takes) != 1:
            raise NotImplementedError('Can only deal with one urban take yet')

        if self.run_config.takes[0] == 'u1_5':
            label_mapping.update({4: 'Dirt'})
        if self.run_config.takes[0] == 'u1_6':
            label_mapping.update({4: 'Metal', 5: 'Dirt'})

        for take in self.run_config.takes:
            labels.append(self.load_labels_from_take_percentage(take))

        try:
            labels = [torch.cat(labels,0)]
        except RuntimeError:
            print("Couldn't stack label tensors due to different sizes!")

        return labels, label_mapping

    def load_labels_from_take_percentage(self, take):
        num_channels = get_class_number(self.run_config.input_type)
        labels = loadmat(f'takes/{take}/labels.mat')['A'].reshape((num_channels,307,307))
        labels = np.argmax(labels, axis=0)

        final_labels = []
        for i in range(num_channels):
            label_copy = labels.copy()
            mask = label_copy == i
            label_copy[mask] = 1
            label_copy[np.logical_not(mask)] = 0
            if self.run_config.label_percentage != 1.0:
                label_copy = self._random_label_selection(label_copy)
            final_labels.append(label_copy)
        final_labels = torch.Tensor(final_labels).unsqueeze(0)
        return final_labels

    def load_labels_from_take(self, takes):
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

    def _random_label_selection(self, labels):
        percentage = self.run_config.label_percentage
        mask = np.where(labels == 1)
        size = int(mask[0].size * percentage)
        selection = np.random.choice(range(0,mask[0].size),size)
        label_selection = np.zeros_like(labels)
        label_selection[(mask[0][selection], mask[1][selection])] = 1
        return label_selection
