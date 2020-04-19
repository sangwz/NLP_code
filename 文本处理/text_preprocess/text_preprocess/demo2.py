# 导入用于对象保存于加载的包
from sklearn.externals import joblib
# 将之前已经训练好的词汇映射器加载进来
t = joblib.load("./Tokenizer")

token = "李宗盛"
# 从词汇映射器中得到李宗盛的index
token_index = t.texts_to_sequences([token])[0][0] - 1
# 初始化一个全零的向量
zero_list = [0] * 6
zero_list[token_index] = 1
print(token, "的one-hot编码为:", zero_list)

