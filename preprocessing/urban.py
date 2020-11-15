import torch
import numpy as np
from scipy.io import loadmat


def load_file(self, file):
    data = loadmat(file)
    hyperspectral_data = data['Y'].reshape((162, 307, 307))
    return torch.from_numpy(hyperspectral_data.astype('int16')).unsqueeze(0)


def load_labels_from_take(self, file, label_amount):
    labels = loadmat(file)['A'].reshape((label_amount, 307, 307))
    labels = np.argmax(labels, axis=0)

    final_labels = []
    for i in range(label_amount):
        label_copy = labels.copy()
        mask = label_copy == i
        label_copy[mask] = 1
        label_copy[np.logical_not(mask)] = 0
        final_labels.append(label_copy)
    final_labels = torch.Tensor(final_labels).unsqueeze(0)
    return final_labels
