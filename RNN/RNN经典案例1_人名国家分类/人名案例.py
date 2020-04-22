from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
# 用于获得常见字母及字符规范化
import string
import unicodedata
# 导入随机工具random
import random
# 导入时间和数学工具包
import time
import math
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包
import matplotlib.pyplot as plt

# 获取所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + ".,;"
# print(all_letters)
n_letters = len(all_letters)


# 字符规范化，unicode转ascii函数
# 关于编码问题我们暂且不去考虑
# 我们认为这个函数的作用就是去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicodedata.normalize('NFD', "Ślusàrski"))
# s = "Ślusàrski"
# a = unicodeToAscii(s)
# print(a)

# 构建一个从持久化文件中读取内容到内存的函数
data_path = '../data/names/'

def readlines(filename):
    try:
        lines = open(filename, encoding="utf8").read().strip().split("\n")
        return [unicodeToAscii(line) for line in lines]
        # return lines
    except Exception as e:
        print(e)

#
# lines = readlines("Chinese.txt")
# print(lines)

# 构建人名类别列表与人名对应关系字典
# 构建的category_lines 形如：{"English": ["Lily"],"Chinese":[]}
category_lines = {}
# all_categories 形如：["English".....,"Chinese"]
all_categories = []

# 读取指定路径下文件，使用glob，path中可以使用正则表达式
for filename in glob.glob(data_path + "*.txt"):
    # 获取每个文件的文件名，就是对应名字的类别
    # print(filename)
    # print(os.path.basename(filename))
    category = os.path.splitext(os.path.basename(filename))[0]
    # print(category)
    all_categories.append(category)
    lines = readlines(filename)
    # 按照对应的类别，将名字列表写入到category_lines 字典中
    category_lines[category] = lines

# 查看类别总数
n_categories = len(all_categories)

# print(n_categories)
# print(category_lines["Italian"][:5])

# 将人名转化为对应onehot张量表示：
def lineToTensor(line):
    # 先构建一个全零的张量,1*n_letters
    tensor = torch.zeros(len(line),1,n_letters)
    for li ,letter in enumerate(line):
        # 找到letter字母在all_letters中的位置，
        # index_letter = all_letters.index(letter)
        # print(index_letter)
        index_letter = all_letters.find(letter)
        print(index_letter)
        tensor[li][0][index_letter] = 1
    return tensor

line = "Bai"
line_tensor = lineToTensor(line)
print("line_tensot:", line_tensor)

# 构建RNN模型

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        """
        初始化函数中有四个函数，分别代表RNN输入最后一位尺寸，RNN隐层最后一维尺寸，RNN层数
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param num_layers:
        """
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.outpu_size = output_size
        self.num_layers = num_layers

        # 实例化预定义的RNN，注意三个参数
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        # 是实例化Liner，将rnn的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的softmax层，用于输出层获取的到类别结果
        # todo : softmax为什么用LogSoftmax而不是用Softmax
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input, hidden):
        # input
        input = input.unsqueeze(0)
        rr ,hn = self.rnn(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        # 初始化函数与rnn相同
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers)
        # 实例化线性链接层
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化softmax层，用于输出得到分类结果
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, hidden, c):
        """
        主逻辑函数中多出一个参数c，也就是lstm中的细胞状态张量
        :param input:
        :param hidden:
        :param c:
        :return:
        """
        # 对input，使用unsqueeze扩展一维张量
        input = input.unsqueeze(dim=0)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        linear = self.linear(rr)
        softmax = self.softmax(linear)
        return softmax, hn, c
    def initHiddenAndC(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


    # GRU模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(self.input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input, hidden):
        # 输入维度进行扩展
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        linear_res = self.linear(rr)
        softmax_res = self.softmax(linear_res)
        return softmax_res, hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# 因为是onehot编码, 输入张量最后一维的尺寸就是n_letters
input_size = n_letters

# 定义隐层的最后一维尺寸大小
n_hidden = 128

# 输出尺寸为语言类别总数n_categories
output_size = n_categories
input = lineToTensor('B').squeeze(0)
print('输入的名字是表示成为张量是----->\n',input)
print('张量的维度是  ---------------->',input.shape)

# 初始化一个三维的隐层0张量, 也是初始的细胞状态张量
hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(n_letters, n_hidden, n_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)
gru = GRU(n_letters, n_hidden, n_categories)

rnn_output, next_hidden = rnn(input, hidden)
print("rnn:", rnn_output)
lstm_output, next_hidden, c = lstm(input, hidden, c)
print("lstm:", lstm_output)
gru_output, next_hidden = gru(input, hidden)
print("gru:", gru_output)


