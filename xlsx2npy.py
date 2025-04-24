import glob

import numpy as np
import pandas as pd
import os

from tqdm import tqdm

if __name__ == '__main__':
    data_path = './PTBXL/'
    xlsx_files = glob.glob(os.path.join(data_path, '*/*.xlsx'))

    sub_folders = os.listdir(data_path)

    npy_path = './npy_files'
    for sub_folder in sub_folders:
        if not os.path.exists(os.path.join(npy_path, sub_folder)):
            os.makedirs(os.path.join(npy_path, sub_folder), exist_ok=True)

    label_dict = {'ALMI': 0,
                  'AMI': 1,
                  'ASMI': 2,
                  'ILMI': 3,
                  'IMI': 4,
                  'LMI': 5,
                  'NORM': 6,
                  'PMI': 7}

    for xlsx_file in tqdm(xlsx_files):
        ecg_data = pd.read_excel(xlsx_file).values
        ecg_data = ecg_data.T

        label = xlsx_file.split('\\')[-2]

        data_dict = {'data': ecg_data,
                     'label': label_dict[label]}

        np.save(os.path.join(os.path.join(npy_path, label + f"/{os.path.basename(xlsx_file).replace('xlsx', 'npy')}")),
                data_dict, allow_pickle=True)
