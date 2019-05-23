# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:41:17 2019

@author: sbtithzy
"""

# 导入需要的包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据集
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# 文本清洗
import re
import nltk##词根
nltk.download('stopwords')###虚词的过滤包（一维）
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])###使用正则只保留字母，用空格代替去掉的部分
    review = review.lower()####将大写变小写
    review = review.split()###将str变成list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]####去掉虚词，保留词根模型
    review = ' '.join(review)###将list转化为str，用空格连接
    corpus.append(review)###生成新的list

# 创建词袋
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)###保留出现次数最多的1500个词根
X = cv.fit_transform(corpus).toarray()####转为矩阵形式，自变量X
y = dataset.iloc[:, 1].values####因变量y

# 训练集和测试集的划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# 使用窄带贝叶斯模型
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# 预测训练集
y_pred = classifier.predict(X_test)

# 生成混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
