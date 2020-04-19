from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 设置训练迭代次数
n_iters =500000
# 设置结果的打印间隔
print_every = 100000
# 设置绘制损失曲线上的制图间隔
plot_every = 50000
# 定义隐层的最后一维尺寸大小
n_hidden = 64

all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s)
                    if unicodedata.category(c) != 'Mn'
                    and c in all_letters)


# 构建一个从持久化文件中读取内容到内存的函数
data_path = './data/names/'
def readLines(filename):
    # 打开指定的文件并读取所有内容，使用strip去除两侧的空白符，然后，以‘、n’进行切分
    lines = open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 构建人名类别列表与人名对应关系字典
category_liens = {}

# 构建所有类别的列表
all_categories = []

# 遍历所有文件，使用glob.glob中利用正则表达式的遍历
for filename in glob.glob((data_path + "*.txt")):
    category = os.path.splitext((os.path.basename(filename)))[0]
    # 逐一将其装入所有类别的列表中
    all_categories.append(category)
    # 读取文件内容，行成名字列表
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入到category_lines字典中
    category_liens[category] = lines

n_categories = len(all_categories)
# print("n_categories:",n_categories)

#
def lineToTensor(line):
    # 首先要初始化一个全零的张量，形状是len(line),1,n_letters
    # 代表人名中的每一个字母都用一个（1*n_letters）张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历每个人名中的每个字符，搜索其对应的索引，将索引位置1
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size,hidden_size,num_layers)
        # 实例化全连接层，作用是将RNN的输出维度转换成指定的输出维度
        self.linear = nn.Linear(hidden_size,output_size)
        # softmax作用：输出层中获取类别结果
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self,input1, hidden):
        # input嗲表人名分类器中的输入张量，1 n_letter
        # hidden 隐藏层张量，相撞numlayers*1*self.hidden_size
        # 注意： 输入到rnn中的张量是三维张量，所以用unsqueeze()函数扩充维度
        input1 = input1.unsqueeze(0)
        # 将input1和hidden输入到RNN实例化对象中，如果num_layers = 1，rr=hn
        rr, hn = self.rnn(input1, hidden)
        # 将从rnn中获取的结果，通过线性层变换和softmax层的处理，最终返回结果
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        # 本函数的作用，用来初始化一个全零的隐藏层张量，维度是3
        return torch.zeros(self.num_layers, 1,  self.hidden_size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的参数与传统RNN相同"""
        super(LSTM, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden, c):
        """在主要逻辑函数中多出一个参数c, 也就是LSTM中的细胞状态张量"""
        # 使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input, hidden以及初始化的c传入lstm中
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # 最后返回处理后的rr, hn, c
        return self.softmax(self.linear(rr)), hn, c

    def initHiddenAndC(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

#因为是onehot编码, 输入张量最后一维的尺寸就是n_letters
input_size = n_letters



# 输出尺寸为语言类别总数n_categories
output_size = n_categories

# num_layer使用默认值, num_layers = 1
input = lineToTensor('B').squeeze(0)

# 初始化一个三维的隐层0张量, 也是初始的细胞状态张量
hidden = c = torch.zeros(1, 1, n_hidden)
rnn = RNN(n_letters, n_hidden, n_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)
gru = GRU(n_letters, n_hidden, n_categories)

rnn_output, next_hidden = rnn(input, hidden)
# print("rnn:", rnn_output)
lstm_output, next_hidden, c = lstm(input, hidden, c)
# print("lstm:", lstm_output)
gru_output, next_hidden = gru(input, hidden)
# print("gru:", gru_output)


# 第四步：构建训练函数并进行训练
# 从输出结果中获取指定类别函数
def categoryFromOutput(output):
    """
    从输出结果中获得指定类别，参数为输出张量output
    :param output:
    :return:
    """
    # 从输出张量中获取最大的值和索引对象，
    # print("output------>",output,output.shape)
    # print("output[0]",output[0])
    top_n, top_i = output.topk(3)
    # print("top_n",top_n)
    # print("top_i",top_i)
    # top_i 为获得的索引值
    # category_i = top_i[0,0,0].item()
    category_i = top_i[0][0][0].item()
    # category_i = top_i[0].item()

    # todo: 课件中是这样写的：category_i = top_i[0].item()
    # top_n tensor([[[-2.8131]]], grad_fn=<TopkBackward>)
    # top_i tensor([[[14]]])，top_i[0]还是top_i[0,0,0]
    # print("category_i--------->",category_i)
    # 根据索引值获得对应语言类别，返回语言类别和索引值
    print()
    return all_categories[category_i], category_i

output = gru_output

category, category_i = categoryFromOutput(output)
# print("category:", category)
# print("category_i:", category_i)

# 随机生成训练数据
def randomTrainingExample():
    """
    该函数用于随机生成训练数据
    :return:
    """
    category = random.choice(all_categories)
    line = random.choice(category_liens[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss()
learning_rate = 0.001

# def trainRNN(category_tensor, line_tensor):
#     """
#     定义训练函数，它的两个参数是category_tensor类别的张量表示，相当于训练数据的标签
#     line_tensor 名字的张量表示，相当于对应训练数据
#     :param category_tensor:
#     :param line_tensor:
#     :return:
#     """
#     # 在函数中，首先通过实例化对象rnn初始化隐层张量
#     hidden = rnn.initHidden()
#
#     # 梯度归零
#     rnn.zero_grad()
#     # print("line_tensor: ------->", line_tensor)
#     # 开始训练，将数据line_tensor的每个字符逐个传入rnn之中，得到最终的结果
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#
#     loss = criterion(output.squeeze(0), category_tensor)
#
#     loss.backward()
#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)
#
#     return output, loss.item()
optimizer1 = torch.optim.Adam(rnn.parameters(), lr = 0.001)
def trainRNN(category_tensor, line_tensor):
    """定义训练函数, 它的两个参数是category_tensor类别的张量表示, 相当于训练数据的标签,
       line_tensor名字的张量表示, 相当于对应训练数据"""

    # 在函数中, 首先通过实例化对象rnn初始化隐层张量
    hidden = rnn.initHidden()

    # 然后将模型结构中的梯度归0
    # rnn.zero_grad()
    optimizer1.zero_grad()
    # 下面开始进行训练, 将训练数据line_tensor的每个字符逐个传入rnn之中, 得到最终结果
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
    # 进行对比计算损失, 需要减少第一个维度, 这里使用squeeze()方法
    loss = criterion(output.squeeze(0), category_tensor)

    # 损失进行反向传播
    loss.backward()
    # 更新模型中所有的参数
    # for p in rnn.parameters():
    #     # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
    #     p.data.add_(-learning_rate, p.grad.data)
    # 返回结果和损失的值
    optimizer1.step()

    return output, loss.item()
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr = 0.001)
def trainLSTM(catetory_tensor, line_tensor):
    """

    :param catetory_tensor: 标签
    :param line_tensor:
    :return:
    """
    # 隐藏层初始化
    h0, c = lstm.initHiddenAndC()
    # 梯度清零
    optimizer_lstm.zero_grad()
    # 输入模型，得到结果
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], h0, c)
    # 打印下outpu的维度信息
    # print("输出的维度是：",output.shape)
    # 计算损失
    loss = criterion(output.squeeze(0), catetory_tensor)
    # 反向传播
    loss.backward()
    # 梯度更新
    # for p in lstm.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    optimizer_lstm.step()

    return output, loss.item()
optimizer_gru = torch.optim.Adam(gru.parameters(), lr = 0.005)
def trainGRU(category_label, category_line):
    """

    :param category_label: 标签值
    :param category_line:  输入的数据
    :return:
    """
    # 初始化hidden
    hidden = gru.initHidden()
    # 梯度清零
    optimizer_gru.zero_grad()
    # 获取输出值
    for i in range(category_line.size()[0]):
        output, hidden = gru(category_line[i], hidden)

    # 计算损失
    # print(" GRU - output------------->%s,\n output-squeeze-------->%s"%(output, output.squeeze(0)))
    loss = criterion(output.squeeze(0), category_label)
    # 反向传播
    loss.backward()
    # 更新梯度
    # for p in gru.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    optimizer_gru.step()

    return output, loss.item()

def timeSince(since):
    """
    获得每次打印的训练耗时，since是训练开始时间
    :param since:
    :return:
    """
    now = time.time()
    # 获得时间差
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

# since = time.time() - 10*60
#
# period = timeSince(since)
# print(period)


# 构建训练过程的日志打印函数


# def train(train_type_fn):
#     """
#     训练过程的日志打印函数，参数train_type_fn代表选择哪种模型训练函数
#     :param train_type_fn:
#     :return:
#     """
#     # 每个制图间隔损失保存列表
#     all_losses = []
#     # 获得训练开始时间戳
#     start = time.time()
#     current_loss = 0
    # 初始间隔损失为0
    # for iter in range(1, n_iters+1):
    #     category, line, category_tensor, line_tensor = randomTrainingExample()
    #     # 将训练数据和对应的张量表示传入到train函数中
    #     output, loss = train_type_fn(category_tensor, line_tensor)
    #     # print("训练输出为：",output)
    #     # 计算制图间隔中的总损失
    #     current_loss += loss
    #     if iter % print_every ==0:
    #         # 通过categoryFromOutput函数获得对应的类别和类别索引
    #         guess, guess_i = categoryFromOutput(output)
    #         # 然后和真是的类别category做比较，如果相同，打对号，不同，叉号
    #         correct = '√' if guess == category else '×(%s)'%category
    #         print('%d %d%% (%s) %.4f %s / %s %s' % (
    #         iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    #
    #     if iter % plot_every == 0:
    #         # 将保存间隔中的平均损失到all_loss 列表中
    #         all_losses.append(current_loss/plot_every)
    #         # 间隔损失重置为0
    #         current_loss = 0
    #
    #     # 返回对应的总损失列表和训练耗时
    # return  all_losses, int(time.time() - start)

def train(train_type_fn):
    """训练过程的日志打印函数, 参数train_type_fn代表选择哪种模型训练函数, 如trainRNN"""
    # 每个制图间隔损失保存列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 从1开始进行训练迭代, 共n_iters次
    for iter in range(1, n_iters + 1):
        # 通过randomTrainingExample函数随机获取一组训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 将训练数据和对应类别的张量表示传入到train函数中
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 计算制图间隔中的总损失
        current_loss += loss
        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)

            # 然后和真实的类别category做比较, 如果相同则打对号, 否则打叉号.
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # 如果迭代数能够整除制图间隔
        if iter % plot_every == 0:
            # 将保存该间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0
    # 返回对应的总损失列表和训练耗时
    return all_losses, int(time.time() - start)
def train_model():
    # 开始训练传统RNN，LSTM，GRU模型，并制作对比图
    all_losses1, period1 = train(trainRNN)
    torch.save(rnn,'./ModelOfName/rnn-adam.pkl')
    # print('rnn_losses--------->', all_losses1)
    # all_losses2, period2 = train(trainLSTM)
    # torch.save(lstm,'./ModelOfName/lstm-sgd.pkl')
    # all_losses3, period3 = train(trainGRU)
    # plt.figure(figsize=(20,8))
    # torch.save(gru,'./ModelOfName/gru-sdg.pkl')
    plt.figure(0)
    # 绘制损失对比曲线
    plt.plot(all_losses1, label="RNN")
    # plt.plot(all_losses2, color="red", label="LSTM")
    # plt.plot(all_losses3, color="orange", label="GRU")
    plt.legend(loc='upper left')


    # 创建画布1
    # plt.figure(1)
    # # x_data=["RNN", "LSTM", "GRU"]
    # # y_data = [period1, period2, period3]
    # x_data=["RNN"]
    # y_data = [period1]
    # # 绘制训练耗时对比柱状图
    # plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.show()


# 构建评估函数
def evaluateRNN(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output.squeeze(0)

def evaluateLSTM(line_tensor):
    hidden ,c = lstm.initHiddenAndC()
    for i in range(line_tensor.size()[0]):
        output, hidden, c  = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)

def evaluateGRU(line_tensor):
    hidden = gru.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)


# 构建预测函数
# 构建预测函数
def predictfileload():
    filename = './data/test_100.csv'
    data = readLines(filename)
    # print(data)
    category_list = []
    category_line_list = []
    # print(data)
    for i in data:
        # print(i,type(i))
        # print((','.split(i)))
        category, category_line = i.split(',')
        category_list.append(category)
        category_line_list.append(category_line)
    # print(category_list,'\n',category_line_list)
    # print(len(category_list),len(category_line_list))
    return category_list, category_line_list
category_test, category_line_test =predictfileload()

def predict(evaluate_fn,input_line, n_predictions=3):
    # print("输入========>",input_line)

    with torch.no_grad():
        output = evaluate_fn(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        for i in range(n_predictions):
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印ouput的值, 和对应的类别
            # print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions中
            predictions.append([value, all_categories[category_index]])

    return predictions

def ratecal():
    pass

rnn_result_list = []
lstm_result_list = []
gru_result_list = []

def predict_test(a):
# for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
    if a == "rnn":
        evaluate_fn = evaluateRNN
    if a == "lstm":
        evaluate_fn = evaluateLSTM
    if a == "gru":
        evaluate_fn = evaluateGRU
    num_correct = 0
    num_correct_3 = 0

    result_list = []
    for i in range(100):

        input_linetensor = category_line_test[i]
        rnn_pre = predict(evaluate_fn,input_linetensor)
        # print(rnn_pre[0][1])
        # print(rnn_pre[1][1])
        result_list.append(rnn_pre[0][1])
        # print(len(result_list),result_list)
        # print('正确的是',category_test[i])
        if rnn_pre[0][1] == category_test[i]:
            num_correct += 1
        if category_test[i] in [rnn_pre[0][1],rnn_pre[1][1],rnn_pre[1][1]]:
            num_correct_3 += 1

    correct_rate = num_correct
    print('%s top1的准确率是：%s%%'%(a,correct_rate))
    print('%s top3的准确率是：%s%%'%(a,num_correct_3))
    return correct_rate, result_list

def predict_value():
    rnn = torch.load('./ModelOfName/rnn-adam.pkl')
    rnn_result_list = predict_test('rnn')
    # lstm = torch.load('./ModelOfName/lstm-sgd.pkl')
    # lstm_result_list = predict_test('lstm')
    # gru = torch.load('./ModelOfName/gru-sdg.pkl')
    # gru_result_list = predict_test('gru')
train_model()
predict_value()

# lstm_result_list = predict_test(evaluateLSTM)
# gru_result_list = predict_test(evaluateGRU)

