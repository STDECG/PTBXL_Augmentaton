import os

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_classification import ECGDataset
from model import se_resnet18


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item() / labels.size(0)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = se_resnet18(num_classes=8).to(device)
    check_points_path = './checkpoints/'
    model.load_state_dict(
        torch.load(os.path.join(check_points_path, 'best-model-classification.pt'), map_location=device))

    test_path = './data/test/'
    test_dataset = ECGDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)

    test_preds, test_trues = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_trues, test_preds)
    test_precision = precision_score(test_trues, test_preds, average='macro')
    test_recall = recall_score(test_trues, test_preds, average='macro')
    test_f1 = f1_score(test_trues, test_preds, average='macro')

    print(f'Test Acc: {test_acc} Test Precision: {test_precision} Test Recall: {test_recall} Test F1: {test_f1}')
