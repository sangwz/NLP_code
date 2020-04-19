import torch.nn as nn
import torch
rnn = nn.LSTM(5, 6, 2)
input = torch.randn(1, 3, 5)
h0 = torch.randn(2,3,6)
c0 = torch.randn(2,3,6)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)
print("hn",hn)
print("cn",cn)
