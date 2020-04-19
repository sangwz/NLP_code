# 1. pytorch

## 1.1 pytorch基础知识

### 1.1.1 pytorch创建张量

​	**已有数据中创建张量**

* 列表中创建

  torch.tensor( list )

* 使用numpy中的数组创建tensor

  torch.tensor(np.array(list))

**创建固定张量**

1. `torch.ones([3,4])` 创建3行4列的**全为1**的tensor
2. `torch.zeros([3,4])`创建3行4列的**全为0**的tensor
3. `torch.ones_like(tensor)` `torch.zeros_like(tensor)`创建与tensor相同形状和数据类型的值全为1/0的tensor
4. `torch.empty(3,4)`创建3行4列的空的tensor，会用无用数据进行填充(手工填充torch.fill_)

​    **在一定范围内创建序列张量**

1. `torch.arange(start, end, step)` 从start到end以step为步长取值生成序列
2. `torch.linspace(start, end, number_steps)` 从start到end之间等差生成number_steps个数字组成序列
3. `torch.logspace(start, end, number_steps, base=10)`在$base^{start}$到$base^{end}$之间等比生成number_steps个数字组成序列

**创建随机张量**

```
torch.rand([3,4])` 创建3行4列的**随机值**的tensor，随机值的区间是`[0, 1)
```

```
torch.randint(low=0,high=10,size=[3,4])` 创建3行4列的**随机整数**的tensor，随机值的区间是`[low, high)
```

``` 
torch.randn([3,4])  创建3行4列的**随机数**的tensor，随机值的分布式均值为0，方差为1
```

### 1.1.2 tensor的属性

#### 获取tensor中的数据

tensor.item()

#### 转化为numpy数组

z.numpy()

#### 获取形状

tensor.size()

tensor.shape

**获取数据类型**

tensor.dtype

**获取阶数**

x.dim()

### 1.1.3 tensor的修改

**形状改变**

`1. tensor.view((3,4))` 类似numpy中的reshape

`2. tensor.t()` 或`tensor.transpose(dim0, dim1)` 转置

`tensor.permute` 变更tensor的轴（多轴转置）

​		permute（）中的参数为原数据中的轴索引

4.`tensor.unsqueeze(dim)` `tensor.squeeze()`填充或者压缩维度

​		tensor.squeeze() 默认去掉所有长度是1的维度，

​		也可以填入维度的下标，指定去掉某个维度

**修改数据类型**

​	1.创建数据的时候指定类型

​			torch.ones(list, dtype = torch.类型)

​	2.改变已有的tensor的类型

​			a.type(torch.float)

​			a.double()

**数据切片**

​		1.切片方式同numpy

​		2.切片赋值  同numpy

### 1.1.4tensor的常用数学运算

1. tensor.add` `tensor.sub` `tensor.abs` `tensor.mm

​		注意：tensor之间元素级别的数学运算同样适用广播机制。

2. 简单函数运算 `torch.exp` `torch.sin` `torch.cos`

3. 原地操作：tensor.add_` `tensor.sub_` `tensor.abs_

   * 张量的加减绝对值

4. 统计操作

   * ```
     ·tensor.max`, 最大
     `tensor.min`, 最小 
     `tensor.mean`, 平均
     `tensor.median` 中位数 
     `tensor.argmax· 最大索引
     ```

## 1.2 梯度下降核反向传播原理

### 	略，同tensor

## 1.3 pytorch自动求导

### 		1.3.1 前向计算

​		**requires_grad** 在定义张量的时候设置属性requires_grad为True

​		对于pytorch中的一个tensor，如果设置它的属性 .requires_grad`为`True`，那么它将会追踪对于该张量的所		有操作。或者可以理解为，这个tensor是一个参数，后续会被计算梯度，更新该参数。

​		``` requires_grad 属性为True，之后的每次计算都会修改其grad_fn属性

​		**grad_fn用来查看反向传播函数的函数类型**

​		**requires_grad_ , 修改tensor张量的梯度属性**

​			 a.requires_grad_(bool)

​		**no_grad**: 

```python
with torch.no_gard():
    c = (a * a).sum()  #tensor(151.6830),此时c没有gard_fn
```

### 1.3.2 梯度计算

API： backword方法进行反向传播计算梯度

​			out.backward()  根据损失函数，对参数计算梯度，注意此时并未更新梯度

​		**注意点**：

- 在tensor的require_grad=False，tensor.data和tensor等价
- require_grad=True时，tensor.data仅仅是获取tensor中的数据
- tensor.numpy，在require_grad = True时不能够直接转换，需要使用 **tensor.detach().numpy()**

## 1.4 pytorch 完成线性回归

## 1.5 pytorch 基础模型组件

### 1.5.1 nn.Module

1. `__init__`需要调用`super`方法，继承父类的属性和方法
2. `forward`方法必须实现，用来定义我们的网络的向前计算的过程.
   * `nn.Module`定义了`__call__`方法，实现的就是调用`forward`方法，即`Lr`的实例，能够直接被传入参数调用，实际上调用的是`forward`方法并传入参数

### 1.5.2 nn.Sequential

* sequential自动实现了forward函数，sequential（）中的参数为定义的神经网络中的层

* ```python
  model = nn.Sequential(nn.Linear(2,64), nn.Linear(64, 1))
  ```

### 1.5.3 optimizer优化器

优化器类型都是由 **torch.optim**提供，例如

1. `torch.optim.SGD(参数，学习率)`
2. `torch.optim.Adam(参数，学习率)`

* 注意
  * `torch.optim.SGD(参数，学习率)`
  * torch.optim.Adam(参数，学习率)

### 1.5.4 损失函数

1. 均方误差:`nn.MSELoss()`,常用于回归问题
2. 交叉熵损失：`nn.CrossEntropyLoss()`，常用于分类问题

## 1.6 pytorch 基础模型

**构建模型，最重要的有两个步骤：**

1. **找到合适的计算关系，随即初始化参数来拟合输入和输出的关系（前向计算，从输入得到输出）**
2. **选取合适的损失函数和优化器来减小损失（反向传播，得到合适的参数）**

### 1.6.1 模型训练固定代码格式

```python
for i in range(30000):
    out = model(x) #3.1 获取预测值
    loss = criterion(y,out) #3.2 计算损失
    optimizer.zero_grad()  #3.3 梯度归零
    loss.backward() #3.4 计算梯度
```



#### 1.6.1.1 pytorch实现线性回归基本模型代码链接(#链接1)

[pytorch实现线性回归模型代码基础版本 [02-pytorch实现线性回归GPU版本.py](pytorch基础模型实现\02-pytorch实现线性回归GPU版本.py) ](#pytorch基础模型实现/01-pytorch实现线性回归基础版本.py)

[github]([https://github.com/sangwz/NLP-/blob/master/NLP%E5%9F%BA%E7%A1%80%E9%98%B6%E6%AE%B5%E5%AD%A6%E4%B9%A0%E5%86%85%E5%AE%B9%E6%B1%87%E6%80%BB/pytorch%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E5%AE%9E%E7%8E%B0/01-pytorch%E5%AE%9E%E7%8E%B0%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E5%9F%BA%E7%A1%80%E7%89%88%E6%9C%AC.py](https://github.com/sangwz/NLP-/blob/master/NLP基础阶段学习内容汇总/pytorch基础模型实现/01-pytorch实现线性回归基础版本.py))

### 1.6.2 模型评估

``` model.eval() #设置模型为评估模式，即预测模式
model.eval() #设置模型为评估模式，即预测模式
predict = model(x)
predict = predict.data.numpy()
```

* 注意
  * model.train(mode=True)

### 1.6.3 GPU上运行代码

1. 判断GPU是否可用 :torch.cuda.is_available()

   * ``` python
     # 固定代码
     torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     ```

2. 模型参数核input数据转换为cuda支持类型，以便gpu调用训练数据

   * ```python
      model.to(device)
      x_true.to(device)
     ```

3. 在gpu上计算结果为cuda的数据类型，需要转化为numpy或者torch的cpu的tensor类型

   * ```python
     predict = predict.cpu().detach().numpy() 
     # detach()的效果和data的相似，但是detach()是深拷贝，data是取值，是浅拷贝
     ```
     
   * 

# 2. 文本预处理day02

命名实体识别

注意中英文命名实体识别加载的训练模型不相同

##  2.1 jieba使用

### 2.1.1 jieba.cut & jieba.lcut

2.1.1 jieba.cut(content,cut_all=False)

* cut_all = False :  此时为精确模式
* cut_all = True: 此时为全模式
* 结果返回一个生成器对象

2.1.2 jieba.lcut(content，cut_all=bool)

* 精确模式全模式同上
* 结果返回一个列表

2.1.3 jieba.cut_for_search(content)

* **搜索引擎模式分词**
  * 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜 索引擎分词.
* 只有一个文本参数

2.1.4 jieba.lcut_for_search(content)

### 2.1.2 用户自定义词典

**自定义词典的作用**

- 添加自定义词典后, jieba能够准确识别词典中出现的词汇，提升整体的识别准确率.
- 词典格式: 每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒.
- 词典样式如下, 具体词性含义请参照 将该词典存为userdict.txt, 方便之后加载使用.

jieba.load_userdict(filepath)

* 加载用户自定义字典

## 2.2 hanlp 使用hanlp进行命名实体识别

### 2.2.1 命名实体识别

通常我们将人名, 地名, 机构名等专有名词统称命名实体

命名实体识别(Named Entity Recognition，简称NER)就是识别出一段文本中可能存在的命名实体.

api：

```python
# 加载命名实体识别的与训练模型，加载中文模型
recognizer_zh = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# 加载英文模型
recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN))
# 进行命名实体识别
recognizer(content)
# 注意content是列表格式
```

### 2.2.2 词性标注

- - 词性: 语言中对词的一种分类方法，以语法特征为主要依据、兼顾词汇意义对词进行划分的结果, 常见的词性有14种, 如: 名词, 动词, 形容词等.
  - 顾名思义, 词性标注(Part-Of-Speech tagging, 简称POS)就是标注出一段文本中每个词汇的词性.

api：

**使用jieba进行词性标注**

```python
import jieba.posseg as pseg
pseg.lcut(content)
# 结果返回一个装有pair元组的列表，对应词汇以及词性
```

**使用hanlp进行词性标注**

```python
# 加载标注模型，中文
tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
# 加载英文标注模型
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
# 进行标注
tagger(content) # 注意content是列表格式
```



## 2 .3文本张量表示方法

* 模型只能接收张量类型数据
* 将一段文本使用张量进行表示，其中一般将词汇为表示成向量，称作词向量，再由各个词向量按顺序组成矩阵形成文本表示.
* **常用的张量表示方法**
  * one-hot
  * word2vec
  * Word Embedding

### 2.3.1 one-hot编码

* onehot编码

```python
# 导入用于对象保存与加载的joblib
from sklearn.externals import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer
# 假定vocab为语料集所有不同词汇集合
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
# 实例化一个词汇映射器对象
t = Tokenizer(num_words=None, char_level=False)
# 使用映射器拟合现有文本数据
t.fit_on_texts(vocab)

for token in vocab:
    zero_list = [0]*len(vocab)
    # 使用映射器转化现有文本数据, 每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0]
    token_index = t.texts_to_sequences([token])[0][0] - 1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 使用joblib工具保存映射器, 以便之后使用
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)
```

* 编码器使用

```python
# 导入用于对象保存与加载的joblib
# from sklearn.externals import joblib
# 加载之前保存的Tokenizer, 实例化一个t对象
t = joblib.load(tokenizer_path)

# 编码token为"李宗盛"
token = "李宗盛"
# 使用t获得token_index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个zero_list
zero_list = [0]*len(vocab)
# 令zero_List的对应索引为1
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list) 
```



### 2.3.2 Word2Vec

**一种将词汇表示成向量的无监督训练方法**

两种方式

* CBOW：周围的词预测中间的词
  * 参数： 窗口大小

* skipgram模式： 与cbow相反，通过中间的预测周围的

#### fasttext工具实现word2vec

[代码段](# NLP基础阶段学习内容汇总/文本处理/01-使用fasttext工具实现word2vec的训练和使用)

[word2vec代码](http://47.92.175.143:8003/1/#11)

- 第一步: 获取训练数据
  - 原始数据处理
    - wikifil.pl: 清理数据中的标签
- 第二步: 训练词向量
  - 使用fasttext中的**train_unsuperbised()**进行无监督训练，参数是**<u>数据集持久化</u>**文化路径
  - 通过get_word_vector方法来获得指定词汇的词向量
- 第三步: 模型超参数设定
  - `model = fasttext.train_unsupervised('data/fil9', "cbow", dim=300, epoch=1, lr=0.1, thread=8)`
- 第四步: 模型效果检验
  - model.get_nearest_neighbors('music')
- 第五步: 模型的保存与重加载
  - model.save_model(文件名)
  - fasttext.load_model(文件名)

### 2.3.4 word embedding可视化分析

```python
# 导入torch和tensorboard的摘要写入方法
import torch
import json
import fileinput
from torch.utils.tensorboard import SummaryWriter
# 实例化一个摘要写入对象
writer = SummaryWriter()

# 随机初始化一个100x50的矩阵, 认为它是我们已经得到的词嵌入矩阵
# 代表100个词汇, 每个词汇被表示成50维的向量
embedded = torch.randn(100, 50)

# 导入事先准备好的100个中文词汇文件, 形成meta列表原始词汇
meta = list(map(lambda x: x.strip(), fileinput.FileInput("./vocab100.csv")))
writer.add_embedding(embedded, metadata=meta)
writer.close()
```

## 2.4 文本数据分析

### 2.4.1 常用的几种文本数据分析方法

* 标签数量分布
* 句子长度分布
* 词频统计与关键词词云

# 3. 新闻主题分类任务

## 3.1.获取数据

# 5. RNN

## 5.1 RNN（recurrent neural network）

循环神经网络，序列形式输入，序列形式输出

关键字：时间步，也就是每个循环步骤

- RNN的循环机制使模型隐层上一时间步产生的结果, 能够作为当下时间步输入的一部分(当下时间步的输入除了正常的输入外还包括上一步的隐层输出)对当下时间步的输出产生影响.

* 因为RNN结构能够很好利用序列之间的关系, 因此针对自然界具有连续性的输入序列, 如人类的语言, 语音等进行很好的处理, 广泛应用于NLP领域的各项任务, 如文本分类, 情感分析, 意图识别, 机器翻译等.
  *  简单说就是RNN能更好的练习上下文

## 5.2 RNN分类

按照输入和输出的结构进行分类:

- N vs N - RNN
- N vs 1 - RNN
  - 常用在文本分类问题上
- 1 vs N - RNN
  - 将图片生成文字等任务
- N vs M - RNN
  - 机器翻译，阅读理解，文本摘要

按照RNN的内部构造进行分类:

- 传统RNN
- LSTM
- Bi-LSTM
- GRU
- Bi-GRU

## 5.3 RNN按照结构分类介绍

![image-20200414200859777](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414200859777.png)



### 5.3.1 传统的RNN网络

内部计算公式：

![image-20200414210431814](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414210431814.png)

激活函数的作用是帮助调节流经网络的值，将数值压缩在-1和1之间

API：

torch.nn.RNN()

参数：

* input_size:输入张量x中特征维度的大小

* hidden_size:隐层张量h中特征维度的大小

* num_layers:隐含层的数量

* nonlinearity:激活函数的选择，默认是tanh

* ```python
  import torch
  import torch.nn as nn
  # rnn第一个参数是输入特征维度，第二个是隐藏层特征维度，第三个是隐藏层层数
  rnn = nn.RNN(5, 6, 2)
  # 输入，第一个表示输入数据量，第二个表示输入的长度，第三个表示特征维度，即tensor的维度
  input = torch.randn(1, 4, 5)
  # 第一个表示隐藏层的数量，第二个表示sequence lenth，第三个表示隐藏层特征维度
  h0 = torch.randn(2, 4, 6)
  output, hn = rnn(input, h0)
  print(output)
  print(hn)
  ```

* 

#### 5.3.1.2 梯度消失或者爆炸

- - 根据反向传播算法和链式法则, 梯度的计算可以简化为以下公式:



![avatar](http://47.92.175.143:8002/img/RNN25.png)



- 其中sigmoid的导数值域是固定的, 在[0, 0.25]之间, 而一旦公式中的w也小于1, 那么通过这样的公式连乘后, 最终的梯度就会变得非常非常小, 这种现象称作梯度消失. 反之, 如果我们人为的增大w的值, 使其大于1, 那么连乘够就可能造成梯度过大, 称作梯度爆炸

### 5.3.2 LSTM模型

LSTM（Long Short-Term Memory），也称长短时记忆结构，传统RNN的变体，

**能缓解梯度消失或爆炸现象**

![image-20200414213741400](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414213741400.png)

四个部分： 

* 遗忘门
  * ![image-20200414213809680](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414213809680.png)
  * 限制上一层的细胞状态的输入信息量
  * sigmod激活函数
* 输入门
  * ![image-20200414220154545](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414220154545.png)
* 细胞状态
  * ![image-20200414220555708](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414220555708.png)
* 输出门
  * ![image-20200414220635045](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414220635045.png)

### 5.3.3 Bi-LSTM结构

![image-20200414220821814](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414220821814.png)

LSTM结构正反各做一次运算，将结果拼接作为输出

* 正反运算之后，增强了语义关联，同时复杂度也有所提升，相比lstm增加了一倍

API：

torcn.nn.LSTM

实例化对象时的参数：

- input: 输入张量x.
- h0: 初始化的隐层张量h.
- c0: 初始化的细胞状态张量c.

参数：

* input_size : 输入张量x中特征维度的大小

* hidden_size: 隐层张量h中特征维度的大小

* num_layers: 隐含层的数量

- nonlinearity: 激活函数的选择, 默认是tanh.
- bidirectional: 是否选择使用双向LSTM, 如果为True, 则使用; 默认不使用.

### 5.3.4 GRU --门控循环单元

* 有效捕捉长序列之间的语义关联，缓解梯度消失或者梯度爆炸的现象

![image-20200414230500669](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414230500669.png)

* ![image-20200414230641919](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414230641919.png)

**API：**

torch.nn.GRU

**实例化对象参数：**

- input: 输入张量x.
- h0: 初始化的隐层张量h.

**类初始化主要参数：**

- input_size: 输入张量x中特征维度的大小.
- hidden_size: 隐层张量h中特征维度的大小.
- num_layers: 隐含层的数量.
- nonlinearity: 激活函数的选择, 默认是tanh.
- bidirectional: 是否选择使用双向LSTM, 如果为True, 则使用; 默认不使用.

**代码示例：**

```
>>> import torch
>>> import torch.nn as nn
>>> rnn = nn.GRU(5, 6, 2)
>>> input = torch.randn(1, 3, 5)
>>> h0 = torch.randn(2, 3, 6)
>>> output, hn = rnn(input, h0)
>>> output
tensor([[[-0.2097, -2.2225,  0.6204, -0.1745, -0.1749, -0.0460],
         [-0.3820,  0.0465, -0.4798,  0.6837, -0.7894,  0.5173],
         [-0.0184, -0.2758,  1.2482,  0.5514, -0.9165, -0.6667]]],
       grad_fn=<StackBackward>)
>>> hn
tensor([[[ 0.6578, -0.4226, -0.2129, -0.3785,  0.5070,  0.4338],
         [-0.5072,  0.5948,  0.8083,  0.4618,  0.1629, -0.1591],
         [ 0.2430, -0.4981,  0.3846, -0.4252,  0.7191,  0.5420]],

        [[-0.2097, -2.2225,  0.6204, -0.1745, -0.1749, -0.0460],
         [-0.3820,  0.0465, -0.4798,  0.6837, -0.7894,  0.5173],
         [-0.0184, -0.2758,  1.2482,  0.5514, -0.9165, -0.6667]]],
       grad_fn=<StackBackward>)
```

![image-20200414231219046](C:\Users\sang\AppData\Roaming\Typora\typora-user-images\image-20200414231219046.png)

## 5.4 注意力机制

### 5.4.1 注意力计算规则

三个指定输入，QKV，三个相等时，称作自注意力计算规则。结果代表query在key和value作用下的注意力表示

常见的注意力计算规则：

* 区别主要是QK两个之间的运算不同，都乘上V

注意力机制实现步骤:

- 第一步: 根据注意力计算规则, 对Q，K，V进行相应的计算.
- 第二步: 根据第一步采用的计算方法, 如果是拼接方法，则需要将Q与第二步的计算结果再进行拼接, 如果是转置点积, 一般是自注意力, Q与V相同, 则不需要进行与Q的拼接.
- 第三步: 最后为了使整个attention机制按照指定尺寸输出, 使用线性层作用在第二步的结果上做一个线性变换, 得到最终对Q的注意力表示.



## 5.5 人名国家分类示例

### 5.5.1 需求介绍

以一个人名为输入, 使用模型帮助我们判断它最有可能是来自哪一个国家的人名, 这在某些国际化公司的业务中具有重要意义, 在用户注册过程中, 会根据用户填写的名字直接给他分配可能的国家或地区选项, 以及该国家或地区的国旗, 限制手机号码位数等等.

