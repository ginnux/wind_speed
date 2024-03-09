from dataset import WindDataset
from torch.utils.data import DataLoader

train_data, y = DataLoader(WindDataset(), batch_size=32, shuffle=False)
