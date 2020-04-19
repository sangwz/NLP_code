# 导入torch和tensorboard导入进来
import torch
import json
import fileinput
from torch.utils.tensorboard import SummaryWriter

# 实例化一个写入对象
writer = SummaryWriter()

# 随机初始化一个100*5的矩阵, 将其视作已经得到的词嵌入矩阵
embedded = torch.randn(100, 50)

# 导入事先准备好的100个中文词汇文件, 形成meta列表原始词汇
meta = list(map(lambda x: x.strip(), fileinput.FileInput("./vocab100.csv")))
writer.add_embedding(embedded, metadata=meta)
writer.close()

