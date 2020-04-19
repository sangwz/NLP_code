# 从io工具包导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 用于正则表达式
import re
# 用于随机生成数据
import random
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch中预定义的优化方法工具包
from torch import optim
# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' 对持久化文件中的数据进行处理，以满足模型训练要求'''
# 将指定语言中的词汇映射成数值
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

'''
这段代码是为了将训练集中的单词语料添加到字典中，构成单词-索引，索引-单词两种形式的字典
'''
class Lang:
    def __init__(self, name):
        '''
        初始化
        :param name:
        '''
        # 将name传入类中
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇的字典，其中0,1对应的sos和eos已经在里面了
        self.index2word = {0:"SOS", 1:"EOS"}
        # 初始化词汇对应的自然数索引，这里从2开始
        self.n_words = 2

    def addSentence(self, sentence):
        '''
        添加句子函数，将句子转化为对应的数值序列，输入参数sentence是一条句子

        :param sentence:
        :return:
        '''
        # 根据一般国家的语言特性，空格分割单词
        # 对句子进行分割，得到对应的词汇列表
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        '''
        添加词汇函数，即将词汇转化为对应的数值，输入参数word是一个单词
        :param word:
        :return:
        '''
        if word not in self.word2index:
            # 如果不在， 将这个词加入其中，并为它对应一个数值
            self.word2index[word] = self.n_words
            # 同时将它的反转形式加入到self.index2word中
            self.index2word[self.n_words] = word
            self.n_words += 1

'''调试'''
# name = 'eng'
# sentence = 'hello I am Jay'
# engl = Lang(name)
# engl.addSentence(sentence)
# print("word2index:",engl.word2index)
# print("index2word",engl.word2index)
# print("n_words", engl.n_words)
'''-----------------------------------------------'''

"""
字符规范化
"""
# 将unicode转为ascii，我们可以认为是去掉一些语言中的中医标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    '''字符串规范化函数，参数s代表传入的字符串'''
    # 去除
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r'([.!？])+',r' \1',s)
    # 使用正则表达式将字符串中不是大小写字符和正常标点的都替换成空格
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s

'''
字符规范化测试代码
'''
s = "Are you kidding me?"
nsr = normalizeString(s)
print(nsr)


'''将数据加载到内存，并实例化类lang'''
data_path = '../data/eng-fra.txt'
def readLangs(lang1, lang2):
    '''
    读取语言函数，参数lang1是源语言的名字，参数lang2是目标语言的名字
    :param lang1:
    :param lang2:
    :return:
    '''
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，以\t进行再次划分，行称子列表，
    # 也就是语言对
    paris = []





