import time

import torch.nn as nn
import  torch.nn.functional as F
import torch
import torchtext
# 文件读取
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification
from torchtext.datasets.text_classification import *
from torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator
import os
from  torch.utils.data import  DataLoader
load_data_path = "./data"
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)
# train_data = open("./ag_news_csv/train.csv")
# train_label = open()
#
# train_dataset, test_dataset = text_classification.DATASETS()
train_csv_path = "data.tar.gz"

def _setup_datasets( root='.data', ngrams=1, vocab=None, include_unk=False):
    # dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive("data.tar.gz")

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))

train_dataset, test_dataset = _setup_datasets()

BATCH_SIZE = 2

device  = torch.device('cuda' if torch.cuda.is_available() else "cpu")
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        """
                description: 类的初始化函数
                :param vocab_size: 整个语料包含的不同词汇总数
                :param embed_dim: 指定词嵌入的维度
                :param num_class: 文本分类的类别总数
                """
        super().__init__()
        # 实例化embedding层, sparse=True代表每次对该层求解梯度时, 只更新部分权重.
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        # 实例化线性层, 参数分别是embed_dim和num_class.
        self.fc = nn.Linear(embed_dim, 1024)
        # 为各层初始化权重
        self.layer = nn.Sequential(
            self.fc,
            nn.BatchNorm1d(1024),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(1024,num_class)

        self.init_weights()

    def init_weights(self):
        """初始化权重函数"""
        # 指定初始权重的取值范围数
        initrange = 0.5
        # 各层的权重参数都是初始化为均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        # 偏置初始化为0
        self.fc.bias.data.zero_()

    def forward(self, text):
        """param
        text: 文本数值映射后的结果
        :return: 与类别数尺寸相同的张量, 用以判断文本类别
        """
        # 获得embedding的结果embedded
        # >>> embedded.shape
        # (m, 32) 其中m是BATCH_SIZE大小的数据中词汇总数
        embedded = self.embedding(text)
        # 接下来我们需要将(m, 32)转化成(BATCH_SIZE, 32)
        # 以便通过fc层后能计算相应的损失
        # 首先, 我们已知m的值远大于BATCH_SIZE=16,
        # 用m整除BATCH_SIZE, 获得m中共包含c个BATCH_SIZE
        # print("embedded的维度：",embedded.shape)
        c = embedded.size(0) // BATCH_SIZE
        # 之后再从embedded中取c*BATCH_SIZE个向量得到新的embedded
        # 这个新的embedded中的向量个数可以整除BATCH_SIZE
        embedded = embedded[:BATCH_SIZE*c]
        # 因为我们想利用平均池化的方法求embedded中指定行数的列的平均数,
        # 但平均池化方法是作用在行上的, 并且需要3维输入
        # 因此我们对新的embedded进行转置并拓展维度
        embedded = embedded.transpose(1, 0).unsqueeze(0)
        # 然后就是调用平均池化的方法, 并且核的大小为c
        # 即取每c的元素计算一次均值作为结果
        embedded = F.avg_pool1d(embedded, kernel_size=c)
        # 最后，还需要减去新增的维度, 然后转置回去输送给fc层
        # print("emb池化后的维度",embedded.shape)
        result = self.fc(embedded[0].transpose(1, 0))
        # print(result.shape)
        result = self.fc2(result)
        return result


# 获取语料中词汇的总数
VOCAB_SIZE = len(train_dataset.get_vocab())

EMBED_DIM = 32

NUM_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE,EMBED_DIM,NUM_CLASS).to(device)
# model = TextSentiment(VOCAB_SIZE,EMBED_DIM,NUM_CLASS)

# 对数据进行batch处理
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch] ).to(device)
    # label = torch.tensor([entry[0] for entry in batch] )
    text = [entry[1] for entry in batch]
    text = torch.cat(text).to(device)
    # text = torch.cat(text)
    return text, label

# 构建训练和验证函数
def train(train_data):
    train_loss = 0
    train_acc = 0
    print(type(train_data))
    # train_data = train_data[:(len(train_data)//BATCH_SIZE)*BATCH_SIZE]
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    # data.to(device)
    # data = data.cuda()

    # print(len(data))
    for i,(text,cls) in enumerate(data):
        # optimizer.zero_grad()# v 1.0
        output = model(text)
        # print("迭代次数：",i)
        # print("模型输出====>",output.shape)
        # print("模型标签维度：",cls.shape)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        # if i % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
        train_acc += (output.argmax(1) == cls).sum().item()

    scheduler.step()
    return train_loss / len(train_data), train_acc / len(train_data)

def valid(valid_data):
    loss = 0
    acc = 0
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, cls in data:
        with torch.no_grad():
            output = model(text)
            loss = criterion(output,cls)
            loss += loss.item()
            acc += (output.argmax(1)==cls).sum().item()
    return loss/len(valid_data), acc/len(valid_data)

# 第四步模型训练和验证

N_EPOCHS = 100
min_valid_loss = float("inf")
criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
train_len = int(len(train_dataset)*0.95)
# 对训练集长度进行处理，之前batch_size设置，如果长度不能整除，会报batch_size不匹配的错误，
# 此处对数据集长度进行处理，使长度能整除batch_size
train_len = int(train_len//BATCH_SIZE*BATCH_SIZE)
sub_train_, sub_valid_ = random_split(train_dataset,[train_len, len(train_dataset) - train_len])
valid_len = (len(train_dataset) - train_len)
valid_len_work = valid_len//BATCH_SIZE*BATCH_SIZE
_, sub_valid_ = random_split(sub_valid_,[valid_len-valid_len_work,valid_len_work])
# print("测试集数据格式：",type(sub_valid_))
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    # 打印训练和验证耗时，平均损失，平均准确率
    print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


