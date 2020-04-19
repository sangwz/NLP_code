import torch
import torch.nn as nn

# 实例化rnn模型对象
# 第一个参数：输入维度，第二个参数隐层维度，第三个参数：隐层层数
rnn = nn.GRU(5,6,2)
# 输入: 数据量， 数据长度， 数据特征数量
input = torch.randn(2, 3, 5)
h0 = torch.randn()