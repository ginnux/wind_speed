import torch
from torch.utils.data import DataLoader
from dataset import WindDataset, getDataset
from sklearn.preprocessing import MinMaxScaler
from model import ApplyModel
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


class Config:
    data_path = "./data/wind_dataset.csv"
    timestep = 10  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    hidden_size = 128  # 隐层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 2  # gru的层数
    epochs = 1  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0003  # 学习率
    model_name = "gru"  # 模型名称
    save_path = "./{}.pth".format(model_name)  # 最优模型保存路径
    device = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()

scaler = MinMaxScaler()
train_dataset, test_dataset = getDataset(
    transforms=scaler.fit_transform, timestep=config.timestep
)
train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


model = ApplyModel(
    input_size=config.feature_size,
    hidden_size=config.hidden_size,
    output_size=config.output_size,
    num_layers=config.num_layers,
    device=config.device,
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_data)

    for x, y in train_bar:
        x = x.to(config.device)
        y = y.to(config.device)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_description(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_data:
            x = x.to(config.device)
            y = y.to(config.device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y)

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)
        print(f"Model saved at {config.save_path}")

print("Finished Training")


test_data = test_dataset.data
test_data = scaler.inverse_transform(test_data)
input_percent = 0.8
inputs = test_data[: int(len(test_data) * input_percent)]
predicts_length = len(test_data) - len(inputs)


inputs = torch.FloatTensor(inputs).to(config.device)
# inputs [batch, length, feature]
inputs = inputs.reshape(1, -1, 1)
y_pred_list = []

predicted_percent = tqdm(range(predicts_length))
for i in predicted_percent:
    y = model(inputs)
    # y [batch, feature]
    y_pred_list.append(y.cpu().detach().numpy())
    y_to_input = torch.unsqueeze(y, dim=1)
    inputs = torch.cat([inputs, y_to_input], dim=1)


predictions = np.stack(y_pred_list, axis=0)
test_data = test_data.reshape(-1)
predicted_data = predictions.reshape(-1)

linespace = np.arange(len(test_data))

plt.figure(figsize=(12, 8))
plt.plot(linespace, test_data, "b")
plt.plot(linespace, predicted_data, "r")
plt.legend()
plt.show()
