import glob
import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import load_npy

if __name__ == '__main__':
    data_path = './npy_files/'
    add_path = './mae_add/'

    train_path = './data_add/train/'
    test_path = './data_add/test/'
    for i in [train_path, test_path]:
        if not os.path.exists(i):
            os.makedirs(i, exist_ok=True)

    npy_files = glob.glob(os.path.join(data_path, '*/*.npy'))
    add_files = [os.path.join(add_path, add_file) for add_file in os.listdir(add_path)]

    total_files = npy_files + add_files  # 35699

    labels = []
    for npy_file in tqdm(total_files):
        _, label = load_npy(npy_file)
        labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(total_files, labels, test_size=0.2, stratify=labels,
                                                        random_state=42)
    for file in tqdm(X_train):
        if '\\' in file:
            shutil.copy(file, os.path.join(train_path, file.split('\\')[-2] + "_" + os.path.basename(file)))
        else:
            shutil.copy(file, os.path.join(train_path, os.path.basename(file)))

    for file in tqdm(X_test):
        if '\\' in file:
            shutil.copy(file, os.path.join(test_path, file.split('\\')[-2] + "_" + os.path.basename(file)))
        else:
            shutil.copy(file, os.path.join(test_path, os.path.basename(file)))
    print(f'Finishing copy {len(X_train)} files to {train_path}, {len(X_test)} files to {test_path}')
