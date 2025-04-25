import os

import numpy as np
import torch
from tqdm import tqdm

from predict_mae_mask import generate_signal
from unet import UNet
from utils import set_seed, load_npy

if __name__ == '__main__':
    set_seed(42)
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    test_path = './mae_test/'

    add_path = './mae_add'
    if not os.path.exists(add_path):
        os.makedirs(add_path, exist_ok=True)

    model = UNet().to(device)
    model.load_state_dict(torch.load('./checkpoints/best_model.pt', weights_only=True, map_location=device))

    num_counts = 3000

    for subfolder in tqdm(os.listdir(test_path)):
        subfolder_path = os.path.join(test_path, subfolder)
        sub_files = os.listdir(subfolder_path)
        sub_count = len(sub_files)

        add_count = num_counts - sub_count

        if add_count <= 0:
            continue

        sub_add = int(add_count / sub_count)  # 每一个文件运行sub_add次
        for sub_file in sub_files:
            _, label = load_npy(os.path.join(subfolder_path, sub_file))

            j = 0
            while j <= sub_add:
                generated_signal = generate_signal(os.path.join(subfolder_path, sub_file), model)
                generated_name = subfolder + f'_{j}_' + sub_file

                data_dict = {'data': generated_signal,
                             'label': label}

                np.save(os.path.join(os.path.join(add_path, generated_name)), data_dict, allow_pickle=True)
                j += 1
