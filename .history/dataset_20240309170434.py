import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class WindDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath="./data/wind_dataset.csv",
        data=None,
        timestep=1,
    ):
        self.data = data
        self.timestep = timestep
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data) - self.timestep

    def __getitem__(self, idx):
        inputs = self.data[idx : idx + self.timestep]
        target = self.data[idx + self.timestep]
        return inputs, target


def getDataset(
    path="./data/wind_dataset.csv",
    transforms=None,
    timestep=1,
    istest=False,
    test_persent=0.2,
):
    df = pd.read_csv(path, index_col=0)
    data = np.array(df["WIND"]).reshape(-1, 1)
    if transforms is not None:
        data = transforms(data)

    dataset = WindDataset(timestep=5)
    return dataset
