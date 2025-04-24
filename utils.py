import os
import random

import numpy as np
import torch


def load_single_npy(npy_file):
    file = np.load(npy_file, allow_pickle=True)[()]

    data = file['data']
    label = file['label']

    return data, label


def normalize(data):
    return (data - min(data)) / (max(data) - min(data) + 1e-10)


def normalize_channel(data):
    normalized = []
    for i in range(len(data)):
        normalized_data = normalize(data[i])
        normalized.append(normalized_data)
    normalized = np.array(normalized)

    return normalized


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='./best-model-attunet.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.threshold = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.threshold:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Val Loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ...\n')

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
