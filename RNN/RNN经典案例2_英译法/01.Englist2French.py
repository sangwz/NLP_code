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
import matplotlib.pyplot as plt

# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
''' ============================对持久化文件中的数据进行处理，以满足模型训练要求======================'''
# 将指定语言中的词汇映射成数值
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

# --------------------------------------------------------
# 实例化参数，
# hidden_size = 25
# input_size = 20
# --------------------------------------------------------
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
        self.index2word = {0: "SOS", 1: "EOS"}
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


# --------------------
# '''调试'''
# name = 'eng'
# sentence = 'hello I am Jay'
# engl = Lang(name)
# engl.addSentence(sentence)
# print("word2index:",engl.word2index)
# print("index2word",engl.word2index)
# print("n_words", engl.n_words)
# '''-----------------------------------------------'''

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
    s = re.sub(r'([.!？])+', r' \1', s)
    # 使用正则表达式将字符串中不是大小写字符和正常标点的都替换成空格
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


# --------------------------------
# 字符规范化测试代码
# s = "Are you kidding me?"
# nsr = normalizeString(s)
# print(nsr)
# ---------------------------------

'''将数据加载到内存，并实例化类lang'''
data_path = '../data/eng-fra.txt'


def readLangs(lang1, lang2):
    '''
    读取语言函数，参数lang1是源语言的名字，参数lang2是目标语言的名字
    :param lang1:
    :param lang2:
    :return:
    函数最后的结果paris是一个语言对的列表
    '''
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # 对lines列表中的句子进行标准化处理，以\t进行再次划分，行称子列表，
    # 也就是语言对
    # 1.先获取每条语句--->2.对语句进行划分，进行标准化处理------>3.对单词进行处理
    # print("lines--------------------->\n\r",lines[:5])
    # print("\n")
    # paris = [[s for s in l.split('\t') ]for l in lines][:5]
    paris = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # print('paris--------------------->\n\r',paris)
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, paris


# ------------------------------
# [readLangs-测试代码]
# lang1 = 'eng'
# lang2 = 'fra'
# input_lang, output_lang, pairs = readLangs(lang1, lang2)
# print(input_lang)
# print(pairs)
# ----------------------------------

""" 
过滤出符合我们要求的语言对
"""
# 设置组成句子中单词或者标点的最多个数
MAX_LENGTH = 10
# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# ----------函数对输入进行过滤，返回输入是否符合长度要求-------#
def filterPair(p):
    '''
    语言对过滤函数，参数P代表输入的语言对，
    :param p: p[0]代表英语句子，对它进行划分，长度小于最大MAX_LENGTH并且以指定的前缀开头
              p[1]代表法语句子，对它进行划分，长度应小于MAX_LENGTH

    :return: 返回bool值，如果输入符合过滤条件，True
    '''
    # print('p[0]------',p[0])
    result_bool = len(p[0].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes) \
                  and len(p[1].split(' ')) < MAX_LENGTH
    # print(result_bool)
    return result_bool


# --------
def filterPairs(pairs):
    '''
    对多个语言对列表进行过滤，pairs代表语言对组成的列表，简称语言对列表
    :param pairs: 语言对列表，即上面上个步骤获取的信息
    :return: 获取符合过滤条件的语言对
    '''
    return [pair for pair in pairs if filterPair(pair)]


# -------------------------
# 过滤测试
# fpairs = filterPairs(pairs)
# print('前五个是：-------->\n',fpairs[:5])
## 问题：
# --------------------------

'''
对以上数据准备函数进行整合，并使用lang对语言对进行数值映射
'''


def prepareData(lang1, lang2):
    '''
    数据准备函数，完成将所有字符串数据向数值形数据的映射，以及对语言对进行过滤
    :param lang1: 源语言名字
    :param lang2: 目标语言的名字
    :return:
    '''
    # 首先通过read_lang函数获得input_lang,output_lang对象，以及字符串类型的语言对列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    # 对字符串类型的语言对列表进行过滤
    pairs = filterPairs(pairs)
    # 对过滤后的语言对列表进行遍历
    for pair in pairs:
        # 使用input_lang,和output_lang的addSentence方法对其进行数值映射
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        # 返回数值映射后的对象，和过滤后语言对
    return input_lang, output_lang, pairs


'''
对数据进行处理，获得input_lang,output_lang对象和语言对列表
# 得到的结果：pairs是一个二维张量，内部一维张量是各个语言对
'''
input_lang, output_lang, pairs = prepareData('eng', 'fra')
# ------------------------------------------------
# 测试
# print("input_n_words:", input_lang.n_words)
# print('dict',input_lang.word2index)
# print(input_lang.word2index['i'])
# print("output_n_words:", output_lang.n_words)
# print("output_n_words:", output_lang.word2index['i'])
# print(random.choice(pairs))
# ------------------------------------------------

'''
将语言对转化为模型输入需要的张量
'''


def tensorFromSentence(lang, sentence):
    '''
    将文本句子转换为张量，参数lang代表传入dLang的word2index方法找到它对应的索引
    :param lang: 实例化对象
    :param sentence: 要转换的句子
    :return:
    '''
    # 对句子进行分割，遍历每一个词汇，使用word2index方法找到对应的索引
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 加入句子结束标记
    indexes.append(EOS_token)
    # 将数据封装成张量，修改维度，变为n * 1的数据，方便计算
    # todo； 测试其它方法实现维度转换，理解view的含义
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    '''
    实现将语言对转换成张量对
    :param pair:  输入的语言对
    :return:
    '''
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)


# -----------------------------------
# 测试代码
# pair = pairs[0]
# print(pair)
# pair_tensor = tensorsFromPair(pair)
# print(pair_tensor)

# --------------------------------------

'''
=======================第三步：构建基于GRU的编码器和解码器===============================
* 构建基于GRU的编码器
* 编码器结构
    {【input --> embeding-->embedded】+prev_hidden}-->gru={output, hidden}

'''


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        设定输出尺寸与隐层节点数相同，即hiddensize = outputsize， 另外词嵌入维度与此也相同
        :param input_size:  输入尺寸，即源语言的词表大小
        :param hidden_size:  隐层节点数，词嵌入维度，gru输入尺寸
        '''
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # 实例化词嵌入层，词嵌入层的第一个维度表示词汇数量，第二个参数表示词汇转换成的向量的维度
        self.embdedding = nn.Embedding(input_size, hidden_size)
        # 实例化GRU模型
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_, hidden):
        '''
        编码器前向逻辑有两个参数
        :param input:  代表源语言输入的embedding层输入张量
        :param hidden:  代表编码器层GRU的初始隐层张量
        :return:
        '''
        # 将输入张量进行embedding 操作，并使器形状为（1，1，-1），-1代表自动计算维度
        # 理论上，编码器每次只是输入一个词，所以词汇映射后的尺寸应该是[1,embedding]
        # 而这里转换成三维的原因使因为torch中预定义gru必须使用三维张量作为输入，因此我们扩展了维度
        output = self.embdedding(input_).view(1, 1, -1).to(device)
        # 然后将embedding层的输出和传入的初始hidden作为gru的输入传入其中
        # 最终gru输出outpu和对应的隐层张量hidden，返回结果
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ---------------------------------------------------------------
# 测试代码
# input_ = pair_tensor[0][0].to(device)
# hidden = torch.zeros(1, 1, hidden_size).to(device)
# encoder = EncoderRNN(input_size, hidden_size)
# encoder_output, hidden = encoder(input_, hidden)
# print(encoder_output)
# --------------------------------------------------------------

'''
实现解码器类
'''


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''
        参数有两个
        :param output_size: 代表整个解码器的输出尺寸，我们希望得到的指定尺寸即目标语言的词表大小
        :param hidden_size:  解码器GRU 的输入尺寸，他的隐层节点数
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 实例化nn中的Embedding层对象，它的参数output这里表示目标语言的词表大小
        # hidden_size 表示目标语言的词嵌入维度
        # todo : 需要再理解下参数
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 实例化GRU对象，输入参数都是hidden_size, 代表它的输入尺寸和隐层节点数相同
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 实例化线性层，对GRU的输出做线性变化，使输出尺寸为output_size
        # 两个参数分别使hidden_size, output_size
        self.out = nn.Linear(hidden_size, output_size)
        # 使用softmax进行处理，以便于分类
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''
        解码器前向逻辑，
        :param input: 代表目标语言的embedding层输入张量
        :param hidden:  hidden代表解码器GRU的初始隐层张量
        :return:
        '''
        # 将输入张量进行embedding操作，使其形状变为（1，1，-1）
        # torch预定义的GRU层只接收三维张量作为输入
        # todo : 此处尝试下其它维度转换方法
        output = self.embedding(input).view(1, 1, -1)
        # 然后使用relu对输出进行处理，可以使embedding层更加稀疏，防止过拟合
        output = F.relu(output)
        # 接下来将embedding的输出以及初始化的hidden张量传入到解码器GRU中
        output, hidden = self.gru(output, hidden)
        # GRU输出的output也是三维张量，第一维没有意义，因此可以使用output[0]进行降维
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ----------------------------------
# 解码器类代码测试
# hidden_size = 25
# output_size = 10
# # 输入参数
# input = pair_tensor[1][0]
# # 初始化第一个隐层张量
# hidden = torch.zeros(1,1,hidden_size)
#
# decoder = DecoderRNN(hidden_size, output_size)
# output, hidden = decoder(input, hidden)
# print(output)
# output--------->tensor([[-2.4240, -2.2452, -2.1819, -2.3925, -2.3496, -2.3927, -2.3774, -2.2828, -2.2195, -2.1976]], grad_fn=<LogSoftmaxBackward>)
# ----------------------------------------------

'''
构建基于GRU和attention的解码器
'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = MAX_LENGTH):
        '''
        :param hidden_size:  代表解码器中GRU的输入尺寸，也是它的隐层节点数
        :param output_size:  代表整个解码器的输出尺寸，也是我们希望得到的指定尺寸，即目标语言的词表大小
        :param dropout_p:  使用dropout时的置0比率，默认0.1
        :param max_length:  代表句子的最大长度
        '''
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 实例化一个enbedding层，输入参数是self.output_size,self.hidden_size
        # todo:弄清楚为甚么输入维度是outpu_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        # 下面时关于注意力的机制的实现方法
        # 根据attention的QKV理论，attention的三个输入参数是Q, K, V
        # 第一步，使用Q和K进行attention权值的计算得到权重矩阵，再与V进行矩阵乘法，得到V的注意力表示结果
        # 常见的三种计算方式（关于注意力的计算方式）
        # 1. 将Q，K进行拼接，得到的结果没经过softmax函数，结果再与V做张量乘法
        # 2. 将Q，K进行纵轴拼接， 做一次线性变化之后再用tanh函数进行激活处理，然后内部求和，
        #       最后使用softmax处理得到结果再与V进行张量乘法，得到结果
        # 3. 将Q和K的转置做点积运算，然后除以一个缩放系数，再使用softmax处理获得结果最后与V做张量乘法
        # 说明：当注意力权重矩阵与V都是三维张量，而且第一维代表batch条数时，做bmm运算
        # 第二步, 根据第一步采用的计算方法, 如果是拼接方法，则需要将Q与第二步的计算结果再进行拼接,
        # 如果是转置点积, 一般是自注意力, Q与V相同, 则不需要进行与Q的拼接.因此第二步的计算方式与第一步采用的全值计算方法有关.
        # 第三步，最后为了使整个attention结构按照指定尺寸输出, 使用线性层作用在第二步的结果上做一个线性变换. 得到最终对Q的注意力表示.
        # 我们这里使用的是第一步中的第一种计算方式, 因此需要一个线性变换的矩阵, 实例化nn.Linear
        # 因为它的输入是Q，K的拼接, 所以输入的第一个参数是self.hidden_size * 2，第二个参数是self.max_length
        # 这里的Q是解码器的Embedding层的输出, K是解码器GRU的隐层输出，因为首次隐层还没有任何输出，会使用编码器的隐层输出
        # 而这里的V是编码器层的输出
        # ................。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        # 接下来实例化另一个线性层，它是attention理论中的第四步的线性层，用于规范输出尺寸
        # 这里它的输入来自第三步的结果, 因为第三步的结果是将Q与第二步的结果进行拼接, 因此输入维度是self.hidden_size * 2
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        # dropout层
        self.dropout = nn.Dropout(self.dropout_p)
        # 实例化GRU，输入输出都是self.hidden_size
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        # 最后实例化gru后面的线性层，也就是我们的解码器输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        :param input:  源数据输入张量
        :param hidden:  初始的隐藏层
        :param encoder_outputs:  解码器的输出张量
        :return:
        '''
        # 根据结构计算图，输入张量进行embedding层并扩展维度
        embeded = self.embedding(input).view(1, 1, -1)
        # 使用dropout 进行随机丢弃，防止过拟合
        embeded = self.dropout(embeded)
        # 进行attention的权重计算，本案例使用第一种计算方式
        # 将QK进行纵轴拼接，做一次线性变换，最后使用softmax进行处理获取结果
        atten_weights = F.softmax(
            self.attn(torch.cat((embeded[0], hidden[0]), 1)), dim=1
        )
        # 然后进行第一步的后半部分，将得到的权重矩阵与V做矩阵乘法运算，当二者都是三维张量，且第一维嗲表batch条数时，做bmm运算
        attn_applied = torch.bmm(atten_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 进行第二步， 通过取【0】用来降维，根据第一步采用的计算方法，需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((embeded[0], attn_applied[0]),1)
        # 第三步， 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        output = self.attn_combine(output).unsqueeze(0)
        # attention结构的结果使用relu激活
        output = F.relu(output)
        # 最后将结果降维并使用softmax处理得到最终的结果
        output = F.log_softmax(self.out(output[0]), dim=1)
        # 返回解码器结果，最后的隐层张量以及注意力权重张量
        return  output, hidden, atten_weights

    def initHidden(self):
        """初始化隐层张量函数"""
        # 将隐层张量初始化成为1x1xself.hidden_size大小的0张量
        return torch.zeros(1, 1, self.hidden_size, device=device)

# -----------------------------------
# 注意力机制类代码测试
#-------------------------------------------

# # 实例化参数
#
# hidden_size = 25
# output_size = 10
#
# # 输入参数
# input = pair_tensor[1][0]
# hidden = torch.zeros(1, 1, hidden_size)
# # encoder_outputs需要是encoder中每一个时间步的输出堆叠而成
# # 它的形状应该是10x25, 我们这里直接随机初始化一个张量
# encoder_outputs  = torch.randn(10, 25)
#
# # 调用
# decoder = AttnDecoderRNN(hidden_size, output_size)
# output, hidden, attn_weights= decoder(input, hidden, encoder_outputs)
# print(output)
# # -------->tensor([[-2.2921, -2.4481, -2.6751, -2.5830, -1.9386, -1.9721, -2.0582, -2.5350,
# #          -2.4884, -2.3541]], grad_fn=<LogSoftmaxBackward>)
# #---------------------------------------------------

'''
==========================第四步， 构建模型训练函数，并进行训练==========================
'''
'''
注意引入了teacher_forcing
teacher_forcing的作用:
能够在训练的时候矫正模型的预测，避免在序列生成的过程中误差进一步放大.
teacher_forcing能够极大的加快模型的收敛速度，令模型训练过程更快更平稳.
'''
'''
4.1 构建训练函数
'''
# 设置teacher_forcing比率为0.5
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
    :param input_tensor: 源语言输入张量
    :param target_tensor:  目标语言输入张量
    :param encoder:  编码器实例对象
    :param decoder:  解码器实例对象
    :param encoder_optimizer: 编码优化方法
    :param decoder_optimizer: 解码器优化方法
    :param criterion: 损失函数计算方法
    :param max_length:句子的最大长度
    :return:
    """

    # -----------------------------encoder编码器训练部分----------------------------
    # 初始化隐层张量
    encoder_hidden = encoder.initHidden()
    # 编码器和解码器优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化编码器输出张量，形状是max_length, encoder.hidden_size 的0 张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 初始设置损失为0
    loss = 0
    # 循环遍历输入张量索引
    for ei in range(input_length):
        # 根据索引从input_tensor取出对应的单词的张量表示，和初始化张量一同传入encoder对象中
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 每次获得的encoder_output是一个3维张量，使用[0, 0]降维变成向量一次存入到encoder——outpus
        # 这样encoder_outpus每一行存的都是对应的句子中每个单词通过解码器的输出结果
        encoder_outputs[ei] = encoder_output[0, 0]

    # -----------------------------decoder解码器部分----------------------------------
    # 初始化解码器的第一个输入，即起始符
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # 初始化解码器的隐层张量，即编码器的隐层输出张量
    decoder_hidden = encoder_hidden
    # 根据随机数与teacher_forcing_ratio对比判断是否需要使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # 如果使用teacher_foricng
    if use_teacher_forcing:
        # 遍历目标张量索引
        for di in range(target_length):
            # 将decoder_input, decoder_hidden, encoder_outputs即attention中的QKV
            # 传入解码器对象，获得decoder_output, decoder_hidden, decoder_attention
            decoder_output, decodder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 因为使用了teacher_forcing，无论解码器输出的decoder_output 是什么，我们都
            # 只使用正确答案，即target_tensor[di]来计算损失
            loss += criterion(decoder_output, target_tensor[di])
            # 并强制将下次的解码器输入设置为正确答案
            decoder_inout = target_tensor[di]

    else:
        # 如果不使用teacher_forcing
        # 仍然遍历目标张量缩影
        for di in range(target_length):
            # 将decoder_input, decoder_hidden, encoder_outputs 传入解码器对象，获得
            # decoder_output, decoder-hidden, decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 不适用teacher_forcing，从decoder_outputs中取答案
            topv, topi = torch.topk(decoder_output, 1)
            # 损失计算仍然使用decoder_output, target_tensor[di]
            loss += criterion(decoder_output, target_tensor[di])
            # 最后如果输入的时终止符，循环停止
            if topi.squeeze().item() == EOS_token:
                break
            # 否则，对topi降维，并分离赋值给decoder_input方便进行下次运算
            # 这里使用了#--detach--# 的分离作用，使这个decoder_input与模型构建的张量图无关，相当于全新的外接输入
            decoder_input = topi.squeeze().detach()

    # 误差反向传播
    loss. backward()
    # 编码器和解码器进行优化，即参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 最后返回平均损失
    return loss.item() / target_length

'''
时间函数构建，用于记录代码运行时间
'''
# 导入时间和数学工具包
import time
import math


def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)


'''
调用训练函数，并打印日志和制图
'''

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """
    训练迭代函数，参数共6个
    :param encoder:  编码器
    :param decoder:  解码器
    :param n_iters:  迭代次数
    :param print_every:  打印间隔
    :param plot_every:  画图间隔
    :param learning_rate:  学习率
    :return:
    """
    # 获得训练开始时间戳
    start = time.time()
    # 每个损失间隔的平均损失保存列表，用于绘制曲线
    plot_losses = []
    # 每个打印日志间隔的总损失，初始为0
    print_loss_total = 0
    # 每个绘制损失间隔的总损失，初始为0
    plot_loss_total = 0
    # 使用预定义的SGD作为优化器，将参数和学习率传入其中
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 损失函数定义
    criterion = nn.NLLLoss()
    # 迭代循环
    for iter in range(1, n_iters + 1):
        # 每次从语言对列表中随机取出一条作为训练语句
        training_pair = tensorsFromPair(random.choice(pairs))
        # 分别从training_pair中取出输入张量和目标张量
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # 通过train获得模型运行的损失
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion)
        # 损失进行累计
        print_loss_total += loss
        plot_loss_total += loss
        # 当迭代步达到日志打印间隔时
        if iter % print_every == 0:
            # 通过总损失除以间隔得到平均损失
            print_loss_avg = print_loss_total / print_every
            # 将总损失归0
            print_loss_total = 0
            # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
            print('%s (%d %d%%) %.4f' % (timeSince(start),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # 当迭代步达到损失绘制间隔时
        if iter % plot_every == 0:
            # 通过总损失除以间隔得到平均损失
            plot_loss_avg = plot_loss_total / plot_every
            # 将平均损失装进plot_losses列表
            plot_losses.append(plot_loss_avg)
            # 总损失归0
            plot_loss_total = 0

    # 绘制损失曲线
    plt.figure()
    plt.plot(plot_losses)
    # 保存到指定路径
    plt.savefig("./s2s_loss.png")

'''
--------------------------------------------------------------------------------------------------
参数设置，模型训练
'''
# 设置隐层大小为256 ，也是词嵌入维度
hidden_size = 256
# 通过input_lang.n_words获取输入词汇总数，与hidden_size一同传入EncoderRNN类中
# 得到编码器对象encoder1
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

# 通过output_lang.n_words获取目标词汇总数，与hidden_size和dropout_p一同传入AttnDecoderRNN类中
# 得到解码器对象attn_decoder1
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# 设置迭代步数
n_iters = 75000
# 设置日志打印间隔
print_every = 5000

#-----------------------------调用
# 调用trainIters进行模型训练，将编码器对象encoder1，码器对象attn_decoder1，迭代步数，日志打印间隔传入其中
trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every)

# -------------------------------------------------------------------------------------------------

'''
==================================第五步， 构建模型评估函数，并进行测试以及Attention效果分析===================================

'''
def evaluate(encoder, decoder, sentence, max_length = MAX_LENGTH):
    """
    模型评估函数
    :param encoder:  编码器
    :param decoder:  解码器
    :param sentence:  需要评估的的句子
    :param max_length: 句子最大长度
    :return:
    """
    # 评估阶段不进行梯度计算
    with torch.no_grad():
        # 对输入的句子进行张量表示
        input_tensor = tensorFromSentence(input_lang, sentence)
        # 获得输入句子的长度
        input_length = input_tensor.size()[0]
        # 初始化编码器隐层张量
        encoder_hidden = encoder.initHidden
        # 初始化编码器输出张量，max_length * encoder.hidden_size的全零张量
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # 遍历循环输入张量索引
        for ei in range(input_length):
            # 根据索引从input_tensor 取出对应的单词的张量表示， 和初始化隐层张量一同传入encoder中
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # 将每次获得的输出encoder_output(三维张量), 使用[0, 0]降两维变成向量依次存入到encoder_outputs
            # 这样encoder_outputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
            encoder_outputs[ei] += encoder_output[0, 0]

        # 初始化解码器的第一个输入，即起始符
        decoder_input = torch.tensor([[SOS_token]], device=device)
        # 初始化解码器的隐层张量即编码器的隐层输出
        decoder_hidden = encoder_hidden
        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(max_length, max_length)
        # 开始循环解码
        for di in range(max_length):
            # 将decoder_input, decoder_hidden, encoder_outputs传入解码器对象
            # 获得decoder_output, decoder_hidden, decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # 取所有的attention结果存入初始化的attention张量中
            decoder_attentions[di] = decoder_attention.data
            # 从解码器输出中获得概率最高的值及其索引对象
            topv, topi = decoder_output.data.topk(1)
            # 从索引对象中取出它的值与结束标志值作对比
            if topi.item() == EOS_token:
                # 如果是结束标志值，则将结束标志装进decoded_words列表，代表翻译结束
                decoded_words.append('<EOS>')
                # 循环退出
                break

            else:
                # 否则，根据索引找到它在输出语言的index2word字典中对应的单词装进decoded_words
                decoded_words.append(output_lang.index2word[topi.item()])

            # 最后将本次预测的索引降维并分离赋值给decoder_input，以便下次进行预测
            decoder_input = topi.squeeze().detach()
        # 返回结果decoded_words， 以及完整注意力张量, 把没有用到的部分切掉
        return decoded_words, decoder_attentions[:di + 1]

'''
随机选择指定数量的数据进行评估：
'''
def evaluateRandomly(encoder, decoder, n=6):
    """随机测试函数, 输入参数encoder, decoder代表编码器和解码器对象，n代表测试数"""
    # 对测试数进行循环
    for i in range(n):
        # 从pairs随机选择语言对
        pair = random.choice(pairs)
        # > 代表输入
        print('>', pair[0])
        # = 代表正确的输出
        print('=', pair[1])
        # 调用evaluate进行预测
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        # 将结果连成句子
        output_sentence = ' '.join(output_words)
        # < 代表模型的输出
        print('<', output_sentence)
        print('')
'''
调用评估函数
'''
# 调用evaluateRandomly进行模型测试，将编码器对象encoder1，码器对象attn_decoder1传入其中
evaluateRandomly(encoder1, attn_decoder1)








