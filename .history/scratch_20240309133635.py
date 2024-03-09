from dataset import WindDataset
from torch.utils.data import DataLoader

train_data = DataLoader(WindDataset(), batch_size=1, shuffle=False)

print(len(train_data))
