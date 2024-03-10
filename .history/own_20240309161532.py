import torch
from dataset import WindDataset, dataset_split
from sklearn.preprocessing import MinMaxScaler


class Config:
    data_path = "./data/wind_dataset.csv"
    timestep = 5  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    hidden_size = 256  # 隐层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 2  # gru的层数
    epochs = 10  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0003  # 学习率
    model_name = "gru"  # 模型名称
    save_path = "./{}.pth".format(model_name)  # 最优模型保存路径


config = Config()

scaler = MinMaxScaler()
dataset = WindDataset(timestep=config.timestep, transforms=scaler.fit_transform)
train_dataset, test_dataset = dataset_split(dataset, train_percent=0.8)

print(len(train_dataset), len(test_dataset))
