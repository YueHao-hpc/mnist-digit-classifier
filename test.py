# train.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleMLP  # ✅ 从 model 文件夹导入模型

# 1. 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. 下载并加载数据
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 3. 初始化模型、损失函数、优化器
model = SimpleMLP()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)