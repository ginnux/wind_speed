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
        transforms=None,
        timestep=1,
        istest=False,
        test_persent=0.2,
    ):
        self.df = pd.read_csv(filepath, index_col=0)
        self.data = np.array(self.df["WIND"]).reshape(-1, 1)
        self.transforms = transforms
        self.timestep = timestep
        if transforms is not None:
            self.data = self.transforms(self.data)

        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data) - self.timestep

    def __getitem__(self, idx):
        inputs = self.data[idx : idx + self.timestep]
        target = self.data[idx + self.timestep]
        return inputs, target


def dataset_split(dataset, train_percent=0.8):
    train_size = int(train_percent * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
