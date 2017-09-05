# coding=utf-8
import os
import os.path
import numpy as np
import codecs
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from sklearn.datasets import fetch_20newsgroups

# extract trainint and test data form sklearn
def extract_data():
    if os.path.exists('./data/Ytrain.npy') and os.path.exists('./data/Ytest.npy') and os.path.exists('./data/texts.pkl'):
        f = codecs.open('./data/texts.pkl', 'rb')
        texts = pickle.load(f)
        f.close()
        Ytrain = np.load('./data/Ytrain.npy')
        Ytest = np.load('./data/Ytest.npy')
        return texts,Ytrain,Ytest
    newsgroups_train = fetch_20newsgroups(subset='train')
    # print(len(newsgroups_train.data))
    trainlabel = np.zeros((len(newsgroups_train.data)), dtype=np.float32)
    cates = newsgroups_train.target_names
    train = []
    count = 0
    index = 0
    for cat in cates:

        newsgroups = fetch_20newsgroups(subset='train', categories=[cat])
        datas = newsgroups.data
        for doc in datas:
            trainlabel[index] = count
            doc = ' '.join(WordPunctTokenizer().tokenize(doc.lower()))
            train.append(doc)
            index += 1
        count += 1
    np.save('./data/Ytrain.npy', trainlabel)

    newsgroups_train = fetch_20newsgroups(subset='test')
    # print(len(newsgroups_train.data))
    testlabel = np.zeros((len(newsgroups_train.data)), dtype=np.float32)
    cates = newsgroups_train.target_names
    test = []
    count = 0
    index = 0
    for cat in cates:
        newsgroups = fetch_20newsgroups(subset='test', categories=[cat])
        datas = newsgroups.data
        for doc in datas:
            testlabel[index] = count
            doc = ' '.join(WordPunctTokenizer().tokenize(doc.lower()))
            test.append(doc)
            index += 1
        count += 1
    np.save('./data/Ytest.npy', testlabel)

    out = codecs.open('./data/texts.pkl', 'wb')
    pickle.dump(train + test, out, 1)
    out.close()
    return train+test, trainlabel, testlabel









