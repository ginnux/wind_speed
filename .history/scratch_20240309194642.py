from dataset import WindDataset
from torch.utils.data import DataLoader
from model import ApplyModel
import torch

net = ApplyModel(
    input_size=1, hidden_size=128, output_size=1, num_layers=2, device="cpu"
)
a = torch.rand(1, 1052, 1)
b = net(a)
print(b.shape)
