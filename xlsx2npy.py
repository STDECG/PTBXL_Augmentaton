import glob

import numpy as np
import pandas as pd
import os

from tqdm import tqdm

if __name__ == '__main__':
    data_path = './PTBXL/'
    xlsx_files = glob.glob(os.path.join(data_path, '*/*.xlsx'))
    print(xlsx_files)

    npy_path = './npy_files'
    if not os.path.exists(npy_path):
        os.makedirs(npy_path, exist_ok=True)

    for xlsx_file in tqdm(xlsx_files):
        ecg_data = pd.rad(xlsx_file)


