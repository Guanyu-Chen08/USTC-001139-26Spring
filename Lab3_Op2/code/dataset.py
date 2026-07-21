import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Subset

class InsectDataset(Dataset):
    def __init__(self, data_source, scaler=None, is_train=True):
        if isinstance(data_source, str):
            with open(data_source, 'r') as f:
                lines = f.readlines()
        else:
            lines = data_source
            
        data, labels = [], []
        for line in lines:
            line = line.strip().replace('(', '').replace(')', '')
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1])])
                labels.append(int(float(parts[2])))
                
        self.X = np.array(data, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int64)
        
        if is_train:
            self.mean, self.std = self.X.mean(axis=0), self.X.std(axis=0)
        else:
            self.mean, self.std = scaler['mean'], scaler['std']
            
        self.X = (self.X - self.mean) / (self.std + 1e-8)
        
    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, idx): 
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloaders(train_path, test_path, batch_size=16, val_split=0.2):
    with open(train_path, 'r') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    val_size = int(len(lines) * val_split)
    train_size = len(lines) - val_size
    
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]
    
    train_ds = InsectDataset(train_lines, is_train=True)
    scaler = {'mean': train_ds.mean, 'std': train_ds.std}
    
    val_ds = InsectDataset(val_lines, scaler=scaler, is_train=False)
    
    test_ds = InsectDataset(test_path, scaler=scaler, is_train=False)
    test_seen = Subset(test_ds, range(0, min(60, len(test_ds))))
    test_unseen = Subset(test_ds, range(min(60, len(test_ds)), len(test_ds)))
    
    full_train_ds = InsectDataset(lines, scaler=scaler, is_train=False)
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test_seen': DataLoader(test_seen, batch_size=batch_size, shuffle=False),
        'test_unseen': DataLoader(test_unseen, batch_size=batch_size, shuffle=False),
        'full_train_loader': DataLoader(full_train_ds, batch_size=batch_size, shuffle=False),
        'full_test_ds': test_ds
    }
    return loaders, scaler