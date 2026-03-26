import torch
from torch import nn


# 自定义神经网络模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    # 前向传播
    def forward(self, x):
        # 第一层
        x = self.linear1(x)
        x = torch.tanh(x)
        # 第二层
        x = self.linear2(x)
        x = torch.relu(x)
        # 第三层
        x = self.out(x)
        y = torch.softmax(x, dim=-1)
        return y


if __name__ == '__main__':
    model = MyModel()

    # 生成训练数据
    x = torch.randn(10, 3)  # 10条数据

    y = model(x)
    # 打印模型预测值
    print("模型预测值：\n", y)
    # 打印模型参数
    print("模型参数：\n", model.state_dict())
