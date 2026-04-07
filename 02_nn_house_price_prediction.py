"""
神经网络——房价预测
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim

print()

# 超参数
batch_size = 64
epoch = 200
lr = 0.1

# 加载数据
df = pd.read_csv("../data/house_prices.csv")
df.drop(['Id'], axis=1, inplace=True)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#   将特征分为数值特征和非数值特征并进行处理
num_features = X.select_dtypes(exclude='object').columns  # 数值类型特征
cat_features = X.select_dtypes(include=['object', 'str']).columns  # 非数值类型特征
num_pipeline = Pipeline([
    ('fillna', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
ct = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features),
])
x_train = ct.fit_transform(x_train)
x_val = ct.transform(x_val)
#   转换成Tensor
x_train = torch.tensor(x_train).float()
x_val = torch.tensor(x_val).float()
y_train = torch.tensor(y_train.values).float()
y_val = torch.tensor(y_val.values).float()

# 创建数据集、数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 定义模型
feature_num = x_train.shape[1]
model = nn.Sequential(
    nn.Linear(feature_num, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
)


# Kaiming初始化参数
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)


model.apply(init_weights)


# 定义损失函数
def log_rmsle(pred, target):
    # 限制预测值范围（从业务上来讲，房价不可能低于1万元，但我感觉强制预测的房价大于1万元也会带来副作用，此处先这么写）
    pred = torch.clamp(pred, 1, float('inf'))
    mse = nn.MSELoss()
    # 关于房价的预测我们更加关心相对误差，因此这里使用对数来衡量误差（当pred接近target时，log(pred/target)才近似等于相对误差）
    return torch.sqrt(mse(torch.log(pred), torch.log(target)))


# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=lr)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 模型训练
for i in range(epoch):
    model.train()
    train_total_loss = 0
    for enter, target in train_loader:
        # 数据迁移
        enter, target = enter.to(device), target.to(device)
        # 前向传播
        output = model(enter)
        # 计算损失
        loss = log_rmsle(output.squeeze(), target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 记录损失
        train_total_loss += loss.item() * enter.shape[0]
    train_loss = train_total_loss / len(train_dataset)
    print(f"Epoch {i + 1:>3}/{epoch:>3}\t train_loss: {train_loss:.4f}")
print('\n······························································\n')

# 模型验证
model.eval()
val_total_loss = 0
for enter, target in val_loader:
    enter, target = enter.to(device), target.to(device)
    output = model(enter)
    loss = log_rmsle(output.squeeze(), target)
    val_total_loss += loss.item() * enter.shape[0]
val_loss = val_total_loss / len(val_dataset)
print(f"val_loss: {val_loss:.4f}")
