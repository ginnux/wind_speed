form data import WindDataset
form torch.utils.data import DataLoader

train_data = DataLoader(WindDataset(), batch_size=32, shuffle=False)