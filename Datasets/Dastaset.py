import os
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 自定义数据集类
class GeoSpatialDataset(Dataset):
    def __init__(self, features, targets, seq_len=10):
        self.features = features.values
        # print(    self.features)
        self.targets = targets
        # print(self.targets)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
