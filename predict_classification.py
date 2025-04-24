import os

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_classification import ECGDataset
from small_model import ECGModel


def evaluate(model, val_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_preds, valid_trues = [], []
    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            valid_loss += loss.item()
            valid_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            valid_trues.extend(labels.detach().cpu().numpy())

    return valid_preds, valid_trues


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ECGModel(num_classes=8).to(device)
    check_points_path = './checkpoints/'
    model.load_state_dict(
        torch.load(os.path.join(check_points_path, 'best-model-classification.pt'), map_location=device))

    test_path = './data/test/'
    test_dataset = ECGDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)

    test_preds, test_trues = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_trues, test_preds)
    print(f'Test Acc: {round(test_acc, 2)}')  # Test Acc: 0.78
