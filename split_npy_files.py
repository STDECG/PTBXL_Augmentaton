import os
import shutil
import numpy as np
from tqdm import tqdm

from utils import set_seed


def split_data(root_path, test_path, train_path, test_ratio=1 / 3):
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    for subfolder in tqdm(os.listdir(root_path)):
        subfolder_path = os.path.join(root_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        os.makedirs(os.path.join(test_path, subfolder), exist_ok=True)
        os.makedirs(os.path.join(train_path, subfolder), exist_ok=True)

        npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]

        num_test = int(len(npy_files) * test_ratio)
        np.random.shuffle(npy_files)

        test_files = npy_files[:num_test]
        train_files = npy_files[num_test:]

        for file in test_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(test_path, subfolder, file))

        for file in train_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(train_path, subfolder, file))


if __name__ == '__main__':
    set_seed(42)

    data_path = './npy_files'
    mae_test = './mae_test'
    mae_train = './mae_train'

    split_data(data_path, mae_test, mae_train)
