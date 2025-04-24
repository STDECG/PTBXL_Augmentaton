import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import load_npy


class ECGDataset(Dataset):
    def __init__(self, root_path):
        self.mask_prob = round(random.uniform(0.1, 0.6), 1)
        self.mask_length = np.random.randint(50, 100)

        self.npy_files = glob.glob(os.path.join(root_path, '*/*.npy'))

    def __getitem__(self, index):
        data, label = load_npy(self.npy_files[index])

        channel_masks = int(len(data) * self.mask_prob)
        mask_channels = random.sample(range(len(data)), k=channel_masks)

        masked_data = data.copy()

        for mask_channel in mask_channels:
            num_masks = np.random.randint(2, 6)
            max_start = len(data[mask_channel]) - self.mask_length
            if max_start <= 0:
                continue

            mask_starts = [np.random.randint(0, max_start)]

            for _ in range(1, num_masks):
                next_start_min = mask_starts[-1] + self.mask_length
                next_start_max = min(mask_starts[-1] + self.mask_length * 2, max_start)
                if next_start_min >= max_start:
                    break
                mask_starts.append(np.random.randint(next_start_min, next_start_max))

            for mask_start in mask_starts:
                masked_data[mask_channel][mask_start:mask_start + self.mask_length] = 0

        data = torch.from_numpy(data).float()
        masked_data = torch.from_numpy(masked_data).float()

        return data, masked_data

    def __len__(self):
        return len(self.npy_files)


if __name__ == '__main__':
    data_path = './mae_train/'
    dataset = ECGDataset(data_path)

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    data, label = next(iter(data_loader))

    print(data.shape)
    print(label.shape)
