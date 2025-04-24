import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import load_npy, normalize_channel


class ECGDataset(Dataset):
    def __init__(self, root_path, mask_prob, mask_length):
        self.data_path = root_path
        self.mask_prob = mask_prob
        self.mask_length = mask_length

        self.npy_files = [npy_file for npy_file in os.listdir(self.data_path) if npy_file.endswith('.npy')]

    def __getitem__(self, index):
        data, label = load_npy(os.path.join(self.data_path, self.npy_files[index]))
        data = normalize_channel(data)

        channel_masks = int(len(data) * self.mask_prob)
        mask_channels = random.sample(range(len(data)), k=channel_masks)

        masked_data = data.copy()

        for mask_channel in mask_channels:
            num_masks = np.random.randint(2, 4)
            mask_starts = [np.random.randint(0, 60)]

            for _ in range(1, num_masks):
                mask_starts.append(
                    np.random.randint(mask_starts[-1] + self.mask_length, mask_starts[-1] + self.mask_length * 2))

            for mask_start in mask_starts:
                masked_data[mask_channel][mask_start:mask_start + self.mask_length] = 0

        data = torch.from_numpy(data).float()
        masked_data = torch.from_numpy(masked_data).float()

        return data, masked_data

    def __len__(self):
        return len(self.npy_files)


if __name__ == '__main__':
    data_path = './npy_files/ALMI/'
    dataset = ECGDataset(data_path, mask_prob=0.5, mask_length=10)

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    data, label = next(iter(data_loader))

    print(data.shape)
    print(label.shape)
