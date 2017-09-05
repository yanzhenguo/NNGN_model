# coding=utf-8
import os
import os.path
import codecs
import nltk
import pickle
import logging
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# def adddocument(path):
#     '''读取文件中的内容，分词后返回内容'''
#     f = codecs.open(path, 'r', 'utf-8')
#     paragraph = f.read().lower()
#     f.close()
#     words = WordPunctTokenizer().tokenize(paragraph)
#     return ' '.join(words)

def extract_data():
    '''提取训练文档和测试文档'''
    if os.path.exists('./data/text.pkl'):
        f = codecs.open('./data/text.pkl', 'rb')
        texts = pickle.load(f)
        f.close()
        return texts
    else:
        raise "not find the ./data/text.pkl file."
        return None
    # rootdir = './data/'
    # texts = []
    # subdirs = ['train/neg', 'train/pos', 'test/neg', 'test/pos']
    # for subdir in subdirs:
    #     for parent, dirnames, filenames in os.walk(rootdir + subdir):
    #         index = 0
    #         for filename in filenames:
    #             content = adddocument(parent + '/' + filename)
    #             texts.append(content)
    # out = codecs.open('./data/text.pkl', 'wb')
    # pickle.dump(texts, out, 1)
    # out.close()
    # return texts


class ImdbCorpus():
    def __init__(self, num_words, max_len, filters=''):
        self.num_words = num_words
        self.max_len = max_len
        self.texts = extract_data()
        tokenizer = Tokenizer(num_words=num_words)
        if filters is not None:
            tokenizer.filters = filters
        tokenizer.fit_on_texts(self.texts[:25000])
        self.tokenizer = tokenizer

    def get_sequence(self):
        return self.tokenizer.texts_to_sequences(self.texts)

    def get_matrix(self):
        return self.tokenizer.texts_to_matrix(self.texts)

    def get_input_bow(self):
        text = self.texts
        xtrain = self.tokenizer.texts_to_matrix(text[:25000])
        xtest = self.tokenizer.texts_to_matrix(text[25000:])
        ytrain = np.zeros((25000,), dtype=np.int8)
        ytest = np.zeros((25000,), dtype=np.int8)
        ytrain[12500:25000] = np.ones((12500,), dtype=np.int8)
        ytest[12500:25000] = np.ones((12500,), dtype=np.int8)
        return [xtrain, ytrain, xtest, ytest]

    def get_sequence_pad(self):
        word_index = self.tokenizer.word_index
        sequences = []
        for i in range(50000):
            t = []
            tokens = self.texts[i].lower().split(' ')
            for j in range(len(tokens)):
                index = word_index.get(tokens[j], 0)
                if index < self.num_words:
                    t.append(index)
                else:
                    t.append(0)
            sequences.append(t)
        return sequences

    def get_input(self):
        sequence = self.get_sequence()
        xtrain = pad_sequences(sequence[0:25000], maxlen=self.max_len)
        xtest = pad_sequences(sequence[25000:50000], maxlen=self.max_len)
        ytrain = np.zeros((25000,), dtype=np.float32)
        ytest = np.zeros((25000,), dtype=np.float32)
        ytrain[12500:25000] = np.ones((12500,), dtype=np.float32)
        ytest[12500:25000] = np.ones((12500,), dtype=np.float32)
        return xtrain, ytrain, xtest, ytest

extract_data()