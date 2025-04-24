import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm

from dataset_mae_mask import ECGDataset
from unet import UNet
from utils import set_seed, EarlyStopping


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for data, noise_data in tqdm(train_loader):
        data = data.to(device)
        noise_data = noise_data.to(device)

        optimizer.zero_grad()
        outputs = model(noise_data)
        loss = criterion(data, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    return train_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, noise_data in tqdm(val_loader):
            data = data.to(device)
            noise_data = noise_data.to(device)
            outputs = model(noise_data)
            loss = criterion(data, outputs)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    check_points_path = './checkpoints/'
    loss_path = './loss_npys/'

    for i in [check_points_path, loss_path]:
        if not os.path.exists(i):
            os.makedirs(i, exist_ok=True)

    train_path = './mae_train/'
    train_dataset = ECGDataset(train_path, mask_prob=0.5, mask_length=30)
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [m - int(0.2 * m), int(0.2 * m)],
                                        generator=torch.Generator().manual_seed(42))

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)

    epochs = 300
    lr = 1e-03
    weigth_decay = 1e-06

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weigth_decay)
    criterion = nn.MSELoss().to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.00001,
                                   path=os.path.join(check_points_path, f'best_model.pt'))

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1} -- Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        np.save(os.path.join(loss_path, f'train_loss'), train_losses,
                allow_pickle=True)
        np.save(os.path.join(loss_path, f'valid_loss'), val_losses,
                allow_pickle=True)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
