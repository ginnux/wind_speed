import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


class WindDataset(torch.utils.data.Dataset):
    def __init__(self, filepath="./data/wind_dataset.csv", transforms=None, timestep=1):
        self.df = pd.read_csv(filepath, index_col=0)
        self.data = np.array(self.df["WIND"]).reshape(-1, 1)
        self.transforms = transforms
        self.timestep = timestep

    def __len__(self):
        return len(self.data) - self.timestep

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.timestep]
        y = self.data[idx + self.timestep]
        return x, y
