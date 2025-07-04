import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    # 这里相当于规定了这个模型的这几个层的数据结构
    def __init__(self):
        super().__init__()
        # 把 28x28 展平成 784 这是输入维度，根据数据 28*28 得到了 实际上 784 个点
        self.flatten = nn.Flatten()  # 这里的flatten用来把这个二维向量给拉平成一维
        # 第一层，全连接，输出 128 维 这是隐藏层的维度，具体选择多少维可以自选，但最好选择 2 的幂作为合适的值，靠近输入层的维度可以稍微大一点，但不能太大，防止过拟合
        self.fc1 = nn.Linear(28 * 28, 128)
        # 第二层，全连接，输出 64 维 也是隐藏层 靠近输出层，维度可以小一点
        self.fc2 = nn.Linear(128, 64)
        # 第三层，输出为10类（0~9）输出层，得到0-9这 10 数，所以为 10 类
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x) 
        x = F.relu(self.fc1(x))                    # ReLU 激活函数用来增强非线性表达
        # 前两次ReLU用于提取特征， ReLU会保留大于0的部分，其他部分被ReLU看成是无用特征，负数直接改成0，不再传播无用特征
        x = F.relu(self.fc2(x))
        # 最后一层不激活，交给 CrossEntropyLoss 处理 softmax
        # 最后一层不激活是因为 最后得到了10个值，表示这10个类别分别对应的概率 其中会有负数，负数意味着模型认为他不很认同这个类别，这个也是有用信息，不能被ReLU直接改为0，所以不用ReLU
        x = self.fc3(x)
        return x
