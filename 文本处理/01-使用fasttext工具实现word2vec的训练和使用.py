# 获取训练数据
# 在这里, 我们将研究英语维基百科的部分网页信息, 它的大小在300M左右
# 这些语料已经被准备好, 我们可以通过Matt Mahoney的网站下载.
# 首先创建一个存储数据的文件夹data
# $ mkdir data
# 使用wget下载数据的zip压缩包, 它将存储在data目录中
# $ wget -c http://mattmahoney.net/dc/enwik9.zip -P data
# 使用unzip解压, 如果你的服务器中还没有unzip命令, 请使用: yum install unzip -y
# 解压后在data目录下会出现enwik9的文件夹
# $ unzip data/enwik9.zip -d data
# 解压数据包到文件夹下，同过head方法查看内容
# 数据处理： perl wikifil.pl data/enwik9 > data/fil9

# 第二步： 训练词向量
import fasttext
model = fasttext.train_unsupervised('data/fil9')

# 查看单词对应的词向量
model.get_word_vector('the')

# 第三步： 模型超参数设定
'''
# 在训练词向量过程中, 我们可以设定很多常用超参数来调节我们的模型效果, 如:
# 无监督训练模式: 'skipgram' 或者 'cbow', 默认为'skipgram', 在实践中，skipgram模
式在利用子词方面比cbow更好.
# 词嵌入维度dim: 默认为100, 但随着语料库的增大, 词嵌入的维度往往也要更大.
# 数据循环次数epoch: 默认为5, 但当你的数据集足够大, 可能不需要那么多次.
# 学习率lr: 默认为0.05, 根据经验, 建议选择[0.01，1]范围内.
# 使用的线程数thread: 默认为12个线程, 一般建议和你的cpu核数相同.
'''
model = fasttext.train_unsupervised('data/fil9','cbow',dim=300,epoch=1,lr=0.1,thread=4)

# 第四步：模型效果检验
model.get_nearest_neighbors('sports')
# 查找"音乐"的邻近单词, 我们可以发现与音乐有关的词汇.
model.get_nearest_neighbors('music')

# 第五步： 模型的保存与加载
model.save_model("fil9.bin")

# 使用fasttext.load_model加载模型
model = fasttext.load_model("fil9.bin")
model.get_word_vector("the")