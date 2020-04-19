import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2,output_size):
        """

        :param query_size: 代表query的最后以维大小
        :param key_size: 代表key的最后一维大小，value_size1嗲表value的倒数第二维大小
        :param value_size1: value的倒数第二维大小
        value = (1, value_size1, value_size2)
        :param value_size2: 代表value的倒数第一维度大小，
        :param output_size: 输出的最后一维大小
        """
        super(Attn, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力机制实现第一步中所需要的线性层
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)
        # 初始化注意力机制实现第三步中所需要的线性层
        self.attn_combine = nn.Linear(self.query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        """
        forward 函数的输入参数有三个，分别时Q， K， V， 根据模型训练常识， 输入给
        atten机制的张量一般时三维张量，这里的三个参数也是三维张量
        :param Q:
        :param K:
        :param V:
        :return:
        """
        cat_ = torch.cat((Q[0], K[0]), 1)
        print("cat_",cat_.shape, cat_)
        weights_inner = self.attn(cat_)
        print("Q;---------",Q[0],Q[0].shape)
        print("K----------",K[0],K.shape)
        print("weights_winner:",weights_inner,weights_inner.shape)
        attn_weights = F.softmax(
            weights_inner, dim=1)

        print("attn_weights:", attn_weights,attn_weights.shape)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)

        out_put = torch.cat((Q[0], attn_applied[0]), 1)
        out_put = self.attn_combine(out_put).unsqueeze(0)
        return out_put, attn_weights


query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size = 64
attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
Q = torch.randn(1,1,32)
K = torch.randn(1,1,32)
V = torch.randn(1,32,64)
out = attn(Q, K ,V)
print(out[0])
print(out[1])