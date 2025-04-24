import os

import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_npy


class ECGDataset(Dataset):
    def __init__(self, data_dir):
        super(ECGDataset, self).__init__()

        self.data_dir = data_dir
        self.npy_files = os.listdir(self.data_dir)

    def __getitem__(self, item):
        data, label = load_npy(os.path.join(self.data_dir, self.npy_files[item]))

        return torch.from_numpy(data).float(), torch.as_tensor(label).type(torch.LongTensor)

    def __len__(self):
        return len(self.npy_files)


if __name__ == '__main__':
    train_path = './data/train/'

    train_set = ECGDataset(train_path)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    data, label = next(iter(train_loader))

    print(data.shape)
    print(label.shape)
