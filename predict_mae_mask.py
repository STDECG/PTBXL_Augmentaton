import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from unet import UNet
from utils import load_npy


def plot_sing_lead(data, outputs):
    plt.figure(figsize=(12, 3), dpi=200)

    plt.plot(data[0], color='blue', label='Original')
    plt.plot(outputs[0], color='green', label='Generated')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def generate_signal(test_file, model):
    data, label = load_npy(test_file)

    mask_prob = round(random.uniform(0.1, 0.6), 1)
    mask_length = np.random.randint(50, 100)

    channel_masks = int(len(data) * mask_prob)
    mask_channels = random.sample(range(len(data)), k=channel_masks)

    masked_data = data.copy()

    for mask_channel in mask_channels:
        num_masks = np.random.randint(2, 6)
        max_start = len(data[mask_channel]) - mask_length
        if max_start <= 0:
            continue

        mask_starts = [np.random.randint(0, max_start)]

        for _ in range(1, num_masks):
            next_start_min = mask_starts[-1] + mask_length
            next_start_max = min(mask_starts[-1] + mask_length * 2, max_start)
            if next_start_min >= max_start:
                break
            mask_starts.append(np.random.randint(next_start_min, next_start_max))

        for mask_start in mask_starts:
            masked_data[mask_channel][mask_start:mask_start + mask_length] = 0

    masked_data_expand = masked_data[np.newaxis, :]
    masked_data_torch = torch.from_numpy(masked_data_expand).float()

    model.eval()
    with torch.no_grad():
        outputs = model(masked_data_torch)
        outputs = outputs.data.cpu().numpy()

    outputs = outputs.squeeze(0)

    return outputs


if __name__ == '__main__':
    device = 'cpu' if torch.cuda.is_available() else 'cpu'

    test_file = './mae_test/ALMI/6.npy'

    model = UNet().to(device)
    model.load_state_dict(torch.load('./checkpoints/best_model.pt', weights_only=True, map_location=device))

    outputs = generate_signal(test_file, model)
    data, label = load_npy(test_file)
    plot_sing_lead(data, outputs)
