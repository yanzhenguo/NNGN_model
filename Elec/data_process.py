# coding=utf-8
import codecs
import os
import pickle
import numpy as np
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def extract_data():
    if os.path.exists('./data/text.pkl'):
        f = codecs.open('./data/text.pkl', 'rb')
        texts = pickle.load(f)
        f.close()
        return texts
    texts = []
    num_train = 25000
    num_test = 25000

    f = codecs.open('./data/elec-25k-train.txt.tok', 'r', 'utf-8')
    for line in f:
        texts.append(line.lower())
    f.close()

    f = codecs.open('./data/elec-test.txt.tok', 'r', 'utf-8')
    for line in f:
        texts.append(line.lower())
    f.close()

    Ytrain = np.zeros((num_train,), dtype=np.int8)
    f = codecs.open('./data/elec-25k-train.cat', 'r')
    index = 0
    for line in f:
        Ytrain[index] = int(line[:-1])
        index += 1
    f.close()

    Ytest = np.zeros((num_test,), dtype=np.int8)
    f = codecs.open('./data/elec-test.cat', 'r')
    index = 0
    for line in f:
        Ytest[index] = int(line[:-1])
        index += 1
    f.close()

    newText = []
    for i in range(num_train):
        if Ytrain[i] == 1:
            newText.append(texts[i])
    for i in range(num_train):
        if Ytrain[i] == 2:
            newText.append(texts[i])
    for i in range(num_train, num_train + num_test):
        if Ytest[i - num_train] == 1:
            newText.append(texts[i])
    for i in range(num_train, num_train + num_test):
        if Ytest[i - num_train] == 2:
            newText.append(texts[i])
    f = codecs.open('./data/text.pkl', 'wb')
    pickle.dump(newText, f, 1)
    f.close()
    return newText

class ElecCorpus():
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