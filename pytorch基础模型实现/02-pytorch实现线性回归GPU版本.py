import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import time

# 1. 定义数据
x = torch.rand([50,1])
y = x*3 + 0.8

#2 .定义模型
class Lr(nn.Module):
    def __init__(self):
        super(Lr,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        out = self.linear(x)
        return out

# 2. 实例化模型，loss，和优化器

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x,y = x.to(device),y.to(device)

model = Lr().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

#3. 训练模型
for i in range(300):
    out = model(x)
    loss = criterion(y,out)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(i,30000,loss.data))

#4. 模型评估
model.eval() #
predict = model(x)
predict = predict.cpu().detach().numpy() #转化为numpy数组
plt.scatter(x.cpu().data.numpy(),y.cpu().data.numpy(),c="r")
plt.plot(x.cpu().data.numpy(),predict,)
plt.show()
