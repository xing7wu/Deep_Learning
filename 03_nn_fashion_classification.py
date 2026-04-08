"""
神经网络——服装分类
"""
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

print()

# 超参数
batch_size = 256
epoch = 20
lr = 0.01

# 加载数据
train_df = pd.read_csv("../data/fashion-mnist_train.csv")
val_df = pd.read_csv("../data/fashion-mnist_val.csv")
x_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
x_val = val_df.iloc[:, 1:].values
y_val = val_df.iloc[:, 0].values
#   转换成Tensor并调整为(N, C, H, W)形状
x_train = torch.tensor(x_train).reshape(-1, 1, 28, 28).float()
y_train = torch.tensor(y_train)
x_val = torch.tensor(x_val).reshape(-1, 1, 28, 28).float()
y_val = torch.tensor(y_val)

# 创建数据集、数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 定义模型
model = nn.Sequential(
    # 卷积层1
    nn.Conv2d(1, 6, 5, 1, 2),
    nn.ReLU(),
    nn.AvgPool2d(2, 2, 0),
    # 卷积层2
    nn.Conv2d(6, 16, 5, 1, 0),
    nn.ReLU(),
    nn.AvgPool2d(2, 2, 0),
    # 拍平
    nn.Flatten(),
    # 线性层1
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    # 线性层2
    nn.Linear(120, 84),
    nn.ReLU(),
    # 线性层3
    nn.Linear(84, 10),
)


# Kaiming初始化参数
def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)


model.apply(init_weights)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# 模型训练
for i in range(epoch):
    model.train()
    train_total_loss = 0
    train_total_acc = 0
    for enter, target in train_loader:
        # 数据迁移
        enter, target = enter.to(device), target.to(device)
        # 前向传播
        output = model(enter)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 记录损失
        train_total_loss += loss.item() * enter.shape[0]
        # 记录准确个数
        y_pred = output.argmax(dim=-1)  # 得到预测分类号
        train_total_acc += y_pred.eq(target).sum().item()
    train_loss = train_total_loss / len(train_dataset)
    train_acc = train_total_acc / len(train_dataset)
    print(f"Epoch {i + 1:>2}/{epoch:>2}\t train_loss: {train_loss:.4f}\t train acc: {train_acc}")
print('\n······························································\n')

# 模型验证
model.eval()
val_total_acc = 0
for enter, target in val_loader:
    enter, target = enter.to(device), target.to(device)
    output = model(enter)
    y_pred = output.argmax(dim=-1)
    val_total_acc += y_pred.eq(target).sum().item()
val_acc = val_total_acc / len(val_dataset)
print(f"val acc: {val_acc}")
