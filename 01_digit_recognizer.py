import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from common.load_data import load_digit_data

# 1. 加载数据
x_train, x_val, y_train, y_val = load_digit_data()

# 2. 创建数据集
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

# 3. 设置超参数
batch_size = 64
epochs = 20
lr = 0.1

# 4. 创建加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 5. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)

# 6. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 7. 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 8. 模型训练和验证
for epoch in range(epochs):
    # 8.1 训练
    model.train()
    train_total_loss = 0
    train_total_acc = 0
    for input, target in train_loader:
        # 数据分批迁移到设备
        input, target = input.to(device), target.to(device)
        # 8.1.1 前向传播，得到输出预测值
        output = model(input)
        # 8.1.2 计算损失
        loss = loss_fn(output, target)
        # 8.1.3 反向传播
        loss.backward()
        # 8.1.4 更新参数
        optimizer.step()
        # 8.1.5 梯度清零
        optimizer.zero_grad()
        # 累加损失
        train_total_loss += loss.item() * input.shape[0]
        # 记录准确个数
        y_pred = output.argmax(dim=-1)  # 得到预测分类号
        train_total_acc += y_pred.eq(target).sum()

    this_train_loss = train_total_loss / len(train_dataset)
    this_train_acc = train_total_acc / len(train_dataset)

    # 8.2 验证（每一epoch训练完成后进行）
    model.eval()
    val_total_loss = 0
    val_total_acc = 0

    with torch.no_grad():
        for input, target in val_loader:
            # 数据分批迁移到设备
            input, target = input.to(device), target.to(device)
            # 8.1.1 前向传播，得到输出预测值
            output = model(input)
            # 8.1.2 计算损失
            loss = loss_fn(output, target)

            # 累加损失
            val_total_loss += loss.item() * input.shape[0]
            # 记录准确个数
            y_pred = output.argmax(dim=-1)  # 得到预测分类号
            val_total_acc += y_pred.eq(target).sum().item()

    this_val_loss = val_total_loss / len(val_dataset)
    this_val_acc = val_total_acc / len(val_dataset)

    print(f"train loss: {this_train_loss}, train acc: {this_train_acc}, val loss: {this_val_loss}, val acc: {this_val_acc}")