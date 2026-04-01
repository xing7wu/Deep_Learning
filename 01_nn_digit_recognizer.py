"""
3层神经网络——手写数字识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

print()

# 设置超参数
batch_size = 64  # 一次反向传播的样本数
epoch = 50  # 数据集完整训练的轮数
lr = 0.1  # 梯度下降的学习率

# 加载数据
df = pd.read_csv("../data/handwritten_digits.csv")
X = df.drop("label", axis=1)
y = df["label"]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#   特征归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_val)
#   转换成Tensor
x_train = torch.tensor(x_train).float()
x_val = torch.tensor(x_test).float()
y_train = torch.tensor(y_train.values)
y_val = torch.tensor(y_val.values)

# 创建数据集、数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 模型训练
for epoch in range(epoch):
    model.train()
    train_total_loss = 0
    train_total_acc = 0
    for input, target in train_loader:
        # 数据迁移
        input, target = input.to(device), target.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 记录损失
        train_total_loss += loss.item() * input.shape[0]
        # 记录准确个数
        train_total_acc += output.argmax(dim=-1).eq(target).sum().item()
    # 得到该epoch的损失和准确率
    this_train_loss = train_total_loss / len(train_dataset)
    this_train_acc = train_total_acc / len(train_dataset)
    print(f"train loss: {this_train_loss}, train acc: {this_train_acc}")
print('\n······························································\n')

# 模型验证
model.eval()
val_total_loss = 0
val_total_acc = 0
for input, target in val_dataset:
    input, target = input.to(device), target.to(device)
    output = model(input)
    loss = loss_fn(output, target)
    val_total_loss += loss.item()
    y_pred = output.argmax(dim=-1)
    val_total_acc += y_pred.eq(target).sum()
this_val_loss = val_total_loss / len(val_dataset)
this_val_acc = val_total_acc / len(val_dataset)
print(f"val loss: {this_val_loss}, val acc: {this_val_acc}")
