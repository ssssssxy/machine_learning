import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)


# step 1 读取数据
# 在本文中，我们使用sklearn中的"20 Newsgroup"数据集
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=(
        'header', 'footers', 'quotes'))
documents = dataset.data
print(len(documents))  # 11314
print(documents[0:10])
print(len(dataset.target_names))


# step 2 数据预处理
# 首先，我们尝试尽可能地清理文本数据。我们的想法是，使用正则表达式replace("[^a-zA-Z#]", " ")一次性删除所有标点符号、数字和特殊字符，这个正则表达式可以替换除带空格的字母之外的所有内容。
# 然后删除较短的单词，因为它们通常并不包含有用的信息。
# 最后，将全部文本变为小写，使得大小写敏感失效。
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
# removing short words
news_df['clean_doc'] = news_df['clean_doc'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
# make all the lowercase
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
print(1)


# step 3 删除停止词
nltk.download('stopwords')
stop_words = stopwords.words('english')
# tokenization
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
# remove stop-words
tokenized_doc = tokenized_doc.apply(
    lambda x: [item for item in x if item not in stop_words])
# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
# 更新df
news_df['clean_doc'] = detokenized_doc


# step 4 文档-词项矩阵（Document-Term Matrix）
# 这是主体建模的第一步。我们将使用sklearn的TfidfVectorizer来创建一个包含1000个词项的文档-词项矩阵。
vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=1000,  # keep top 1000 terms
                             max_df=0.5,
                             smooth_idf=True)
# 参数max_features: 如果不为None，构建一个词汇表，仅考虑max_features–按语料词频排序，如果词汇表不为None，这个参数被忽略。也就是说这个参数可以加快运算速度。
# 我们也可以使用全部词项来创建这个矩阵，但这回需要相当长的计算时间，并占用很多资源。因此，我们将特征的数量限制为1000。
X = vectorizer.fit_transform(news_df['clean_doc'])
print(X.shape)  # check shape of the document-term matrix


# step 5 主题建模
# 下一步是将每个词项和文本表示为向量。我们将使用文本-词项矩阵，并将其分解为多个矩阵。我们将使用sklearn的TruncatedSVD来执行矩阵分解任务。
# 由于数据来自20个不同的新闻组，所以我们打算从文本数据中提取出20个主题。可以使用n_components参数来制定主题数量。
# SVD represent documents and terms in vectors
svd_model = TruncatedSVD(
    n_components=20,
    algorithm='randomized',
    n_iter=100,
    random_state=122)
svd_model.fit(X)
print(len(svd_model.components_))
terms = vectorizer.get_feature_names()
# svd_model的组成部分即是我们的主题，我们可以通过svd_model.components_来访问它们。
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
    print()
    print("Topic " + str(i) + ": ", end=" ")
    for t in sorted_terms:
        print(t[0], end=" ")
        print("", end=" ")


# step 6 主题可视化
# 为了找出主题之间的不同，我们将其可视化。当然，我们无法可视化维度大于3的数据，
# 但有一些诸如PCA和t-SNE等技术可以帮助我们将高维数据可视化为较低维度。
# 在这里，我们将使用一种名为UMAP（Uniform Manifold Approximation and Projection）的相对较新的技术。
X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(
    n_neighbors=150,
    min_dist=0.5,
    random_state=12).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.scatter(embedding[:, 0], embedding[:, 1],
            c=dataset.target,
            s=10,  # size
            edgecolor='none'
            )
plt.show()

print(1)


