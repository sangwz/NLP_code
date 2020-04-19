import torch
import torch.nn as nn
# rnn第一个参数是输入特征维度，第二个是隐藏层特征维度，第三个是隐藏层层数
rnn = nn.RNN(5, 6, 2)
# 输入，第一个表示输入数据量，第二个表示输入的长度，第三个表示特征维度，即tensor的维度
input = torch.randn(1, 4, 5)
# 第一个表示隐藏层的数量，第二个表示sequence lenth，第三个表示隐藏层特征维度
h0 = torch.randn(2, 4, 6)
output, hn = rnn(input, h0)
print(output)
print(hn)