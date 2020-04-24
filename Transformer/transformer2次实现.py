import torch
import torch.nn as nn
import math
import numpy as np
# torch中变量封装函数
from torch.autograd import Variable
import torch.nn.functional as F

# 实现步骤
# 1. 实现embedding层


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """

        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
# 2. 实现positionembedding层

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # pe-->(max_len, d_model)
        # position
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        # 此时，pe是二维张量，需要进行升维处理，才能和输入的三维张量进行运算
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # pe升维度后的第二维，也就是词表长度那维，与x输入可能不会相同，若要进行x，和位置编码相加，会
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad = False)
        return self.dropout(x)


'''==========================编码器实现====================================='''

# 生成掩码张量的代码

def subsequent_mask(size):
    """
    生成掩码张量，函数中
    :param size:
    :return:
    """
    # 掩码张量
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(1- subsequent_mask)

# 注意力机制的实现
def attention(query, key, value, mask=None, dropout=None):
    """
    # todo qkv三个变量的维度是什么
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    # 1. 确定query的最后一维的大小
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# 多头注意力机制实现
# 多头注意力实现需要用到copy模块进行deepcopy，方便创建多个没有关联的相同的模型层


import copy
# clone函数定义


def clones(module, N):
    """
    将模型进行拷贝N次，将
    :param module:  要拷贝的模型
    :param N:  拷贝次数
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 实现多头模型的类


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        多头注意力机制的实现，对输入进行多头的处理
        :param head:
        :param embedding_dim:
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim//head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2) for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head*self.d_k)
        return self.linears[-1](x)

    # 前馈全连接层代码实现

# 3.
# 4.
# 5.








