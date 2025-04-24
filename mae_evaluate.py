import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mae_mask import ECGDataset
from unet import UNet
from utils import set_seed


def calculate_rmse(preds, targets):
    mse = F.mse_loss(preds, targets, reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse.item()


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet().to(device)
    check_points_path = './checkpoints/'
    model.load_state_dict(
        torch.load(os.path.join(check_points_path, 'best_model.pt'), map_location=device, weights_only=True))

    test_path = './mae_test/'
    test_dataset = ECGDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for data, noise_data in tqdm(test_loader):
            data, noise_data = data.to(device), noise_data.to(device)

            pred = model(noise_data)
            batch_rmse = calculate_rmse(pred, data)
            total_rmse += batch_rmse * data.size(0)

    mean_rmse = total_rmse / len(test_loader.dataset)
    print(f"RMSE: {round(mean_rmse, 2)}")
