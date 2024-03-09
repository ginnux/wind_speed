from dataset import WindDataset
from torch.utils.data import DataLoader

train_data = DataLoader(WindDataset(), batch_size=32, shuffle=False)

for x, y in train_data:
    print(x, y)
    break
