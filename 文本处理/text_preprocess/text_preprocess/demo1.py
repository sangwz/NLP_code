# 导入用于对象保存和加载的包
from sklearn.externals import joblib
# 导入keras中的词汇映射器Tokenizer
from keras.preprocessing.text import Tokenizer

# 初始化一个词汇表
vocab = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}

# 实例化一个词汇映射器
t = Tokenizer(num_words=None, char_level=False)

# 在映射器上拟合现有的词汇表
t.fit_on_texts(vocab)

# 循环遍历词汇表, 将每一个单词映射为one-hot张量表示
for token in vocab:
    # 初始化一个全零向量
    zero_list = [0] * len(vocab)
    # 使用映射器转化文本数据, 每个词汇对应从1开始
    token_index = t.texts_to_sequences([token])[0][0] - 1
    # 将对应的位置赋值为1
    zero_list[token_index] = 1
    print(token, "的one-hot编码为:", zero_list)

# 将拟合好的词汇映射器保存起来
tokenizer_path = "./Tokenizer"
joblib.dump(t, tokenizer_path)

