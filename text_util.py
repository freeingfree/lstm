# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:53:05 2018

@author: wuxin
"""
import matplotlib.pyplot as plt
import codecs
import jieba
import pandas as pd
from wordcloud import WordCloud
import itertools
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#载入数据
def load_txt(filename):
    lists = []
    with codecs.open(filename, "r", "utf-8") as f:
        for each in f.readlines():
            if each != "":
                lists.append(each.strip("\n").lower())
    return lists

#jieba分词 使用空格连接分词结果
def seg_words_with_blank(datalist):
    stopwords = codecs.open('stopwords.txt', 'r', 'UTF8').read().split('\r\n')
    processed_texts = []
    for text in datalist:
        words = []
        seg_list = jieba.cut(text)
        for seg in seg_list:
            if (seg.isalpha()) & (seg not in stopwords):
                words.append(seg)
        sentence = " ".join(words)
        processed_texts.append(sentence)
    return processed_texts

#jieba分词 不使用空格连接分词结果
def seg_words(datalist):
    stopwords = codecs.open('stopwords.txt', 'r', 'UTF8').read().split('\r\n')
    processed_texts = []
    for text in datalist:
        words = []
        seg_list = jieba.cut(text)
        for seg in seg_list:
            if (seg.isalpha()) & (seg not in stopwords):
                words.append(seg)
        processed_texts.append(words)
    return processed_texts

#词云 显示
def word_cloud(text):
    wc = WordCloud(
    background_color="white",
    max_words=200,
    font_path="C:\\Windows\\Fonts\\STFANGSO.ttf",
    min_font_size=15,
    max_font_size=50, 
    width=400 
    )
    wordcloud = wc.generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    

#转化为Count型稀疏矩阵 binary为True表示不计数，只用0和1表示单词是否出现
def count_sparse_matrix(texts, binary = False):
    vectorizer = CountVectorizer(binary=binary)
    vectorizer.fit(texts)
    
    vector = vectorizer.transform(texts)
    result = pd.DataFrame(vector.toarray())
    
    keys = []
    values = []
    for key,value in vectorizer.vocabulary_.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame(data = {"key" : keys, "value" : values})
    colnames = df.sort_values("value")["key"].values
    result.columns = colnames
    return result

#转化为Vectorizer型稀疏矩阵
def vectorizer_sparse_matrix(texts, binary = False):
    vectorizer = TfidfVectorizer(binary=binary)
    vectorizer.fit(texts)
    
    vector = vectorizer.transform(texts)
    result = pd.DataFrame(vector.toarray())
    
    keys = []
    values = []
    for key,value in vectorizer.vocabulary_.items():
        keys.append(key)
        values.append(value)
    df = pd.DataFrame(data = {"key" : keys, "value" : values})
    colnames = df.sort_values("value")["key"].values
    result.columns = colnames
    return result

#关键字提取 使用TF-IDF排序 num为提取前几个关键字 
def key_words(texts, num):
    textmatrix = vectorizer_sparse_matrix(texts)
    features = pd.DataFrame(textmatrix.apply(sum, axis=0))
    features.columns, features["NAME"], features.index = ["TF_IDF"], features.index, [i for i in range(features.shape[0])]
    features = features.sort_values(by="TF_IDF", ascending=False)
    keywords = list(features.iloc[0 : num]["NAME"])
    if "" in keywords:
        keywords.remove("")
        keywords.append(features.iloc[num]["NAME"])
    return keywords


#高频词提取 使用计数法排序 num为提取前几个单词
def top_words(texts, num):
    textmatrix = count_sparse_matrix(texts)
    features = pd.DataFrame(textmatrix.apply(sum, axis=0))
    features.columns, features["NAME"], features.index = ["COUNT"], features.index, [i for i in range(features.shape[0])]
    features = features.sort_values(by="COUNT", ascending=False)
    topwords = list(features.iloc[0 : num]["NAME"])
    if "" in topwords:
        topwords.remove("")
        topwords.append(features.iloc[num]["NAME"])
    return topwords

#低频词提取 使用计数发排序 num为提取后几个单词
def last_words(texts, num):
    textmatrix = count_sparse_matrix(texts)
    features = pd.DataFrame(textmatrix.apply(sum, axis=0))
    features.columns, features["NAME"], features.index = ["COUNT"], features.index, [i for i in range(features.shape[0])]
    features = features.sort_values(by="COUNT", ascending=True)
    lastwords = list(features.iloc[0 : num]["NAME"])
    if "" in lastwords:
        lastwords.remove("")
        lastwords.append(features.iloc[num]["NAME"])
    return lastwords

#将稀疏矩阵转变为lists(双层嵌套)类型
def matrix_to_wordlists(textmatrix):
    wordlists = []
    wordnums = []
    for row in textmatrix.iterrows():
        item = row[1]
        words = []
        num = 0
        for word in item.index:
            if item[word] != 0:
                words.append(word)
                num = num + 1
        wordlists.append(words)
        wordnums.append(num)
    return wordlists,wordnums

#平铺多层list为一层words
def wordlists_to_words(wordlists):
    return list(itertools.chain.from_iterable(wordlists))

#为单词提供编码
def word_dict(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int2vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab2int = {word: ii for ii, word in int2vocab.items()}
    return vocab2int, int2vocab