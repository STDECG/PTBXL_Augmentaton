import os
import shutil
import numpy as np
from tqdm import tqdm


def split_data(toot_path, test_path, train_path, test_ratio=1 / 3):
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    for subfolder in tqdm(os.listdir(toot_path)):
        subfolder_path = os.path.join(toot_path, subfolder)

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
    source_folder = './npy_files'
    test_folder = './mae_test'
    train_folder = './mae_train'

    split_data(source_folder, test_folder, train_folder)