# 导入必备的工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jieba

# 设置显示风格
plt.style.use('fivethirtyeight')

# 利用pandas读取训练数据和验证数据
train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")

# 获得训练数据标签的数量分布
# sns.countplot("label", data=train_data)
# plt.title("train_data")
# plt.show()

# 获得验证数据标签的数量分布
# sns.countplot("label", data=valid_data)
# plt.title("valid_data")
# plt.show()

# 在训练数据中添加新的句子长度列, 每个元素的值都是对应句子的长度
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))

# 绘制句子长度列的数量分布
# sns.countplot("sentence_length", data=train_data)
# plt.xticks([])
# plt.show()

# 绘制dist长度分布图
# sns.distplot(train_data["sentence_length"])
# plt.yticks([])
# plt.show()

# 在验证数据中添加新的句子长度列, 每个元素的值对应句子的长度
valid_data["sentence_length"] = list(map(lambda x: len(x), valid_data["sentence"]))

# 绘制句子长度列的数量分布图
# sns.countplot("sentence_length", data=valid_data)
# plt.xticks([])
# plt.show()

# 绘制dist长度分布图
# sns.distplot(valid_data["sentence_length"])
# plt.yticks([])
# plt.show()


# 绘制训练数据语句长度的散点图
# sns.stripplot(y="sentence_length", x="label", data=train_data)
# plt.show()

# 绘制验证数据语句长度的散点图
# sns.stripplot(y="sentence_length", x="label", data=valid_data)
# plt.show()


# 导入jieba 工具包和chain工具包, 用于分词扁平化列表
from itertools import chain

# 进行训练集的句子进行分词, 并统计出不同词汇的总数
# train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
# print("训练集共包含不同词汇总数为:", len(train_vocab))

# 进行验证集的句子进行分词, 并统计出不同词汇的总数
# valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data["sentence"])))
# print("验证集共包含不同词汇总数为:", len(valid_vocab))


# 导入jieba 中的词性标注工具包
import jieba.posseg as pseg

# 定义获取形容词的列表函数
def get_a_list(text):
    # 使用jieba的词性标注方法来切分文本, 获得两个属性word,flag
    # 利用flag属性去判断一个词汇是否是形容词
    r = []
    for g in pseg.lcut(text):
        if g.flag == 'a':
            r.append(g.word)
    return r


# 导入绘制词云的工具包
from wordcloud import WordCloud

# 定义获取词云的函数并绘图
def get_word_cloud(keywords_list):
    # 首先实例化词云类对象, 里面三个参数
    # font_path: 字体路径,为了能够更好的显示中文
    # max_words: 指定词云图像最多可以显示的词汇数量
    # backgroud_color: 代表图片的北京颜色
    wordcloud = WordCloud(max_words=100, background_color='white')

    # 将传入的列表参数转化为字符串形式, 因为词云对象的参数要求是字符串类型
    keywords_string = " ".join(keywords_list)
    # 生成词云
    wordcloud.generate(keywords_string)

    # 绘图
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 获取训练集上的正样本
# p_train_data = train_data[train_data["label"]==1]["sentence"]

# 对正样本的每个句子提取形容词
# train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))

# 获取训练集上的负样本
# n_train_data = train_data[train_data["label"]==0]["sentence"]

# 对负样本的每个句子提取形容词
# train_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_train_data))

# 调用获取词云的函数
# get_word_cloud(train_p_a_vocab)
# get_word_cloud(train_n_a_vocab)


# 获取验证集的数据正样本
p_valid_data = valid_data[valid_data["label"]==1]["sentence"]

# 获取正样本的每个句子的形容词
valid_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_valid_data))

# 获取验证集的数据负样本
n_valid_data = valid_data[valid_data["label"]==0]["sentence"]

# 获取负样本的每个句子的形容词
valid_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_valid_data))

# 调用获取词云的函数
get_word_cloud(valid_p_a_vocab)
get_word_cloud(valid_n_a_vocab)




















































