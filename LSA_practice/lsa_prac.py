import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)

from sklearn.datasets import fetch_20newsgroups

# step 1 读取数据
# 在本文中，我们使用sklearn中的"20 Newsgroup"数据集
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('header','footers','quotes'))
documents = dataset.data
print(len(documents))
print(documents[0:10])
print(len(dataset.target_names))

# step 2 数据预处理
# 首先，我们尝试尽可能地清理文本数据。我们的想法是，使用正则表达式replace("[^a-zA-Z#]", " ")一次性删除所有标点符号、数字和特殊字符，这个正则表达式可以替换除带空格的字母之外的所有内容。
# 然后删除较短的单词，因为它们通常并不包含有用的信息。
# 最后，将全部文本变为小写，使得大小写敏感失效。
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# make all the lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
print(1)

# step 3 删除停止词
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
