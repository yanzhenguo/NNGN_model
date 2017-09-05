# -*- coding: utf-8 -*-
import codecs
import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, Concatenate
import data_process

NUM_WORDS = 30000
MAX_LEN = 300

imdb = data_process.ImdbCorpus(num_words=NUM_WORDS, max_len=MAX_LEN)
data1, Ytrain, data2, Ytest = imdb.get_input()

Xtrain1 = np.zeros((25000, (MAX_LEN - 2) * 3), dtype=np.int)
Xtest1 = np.zeros((25000, (MAX_LEN - 2) * 3), dtype=np.int)
for i in range(25000):
    for j in range(MAX_LEN - 2):
        Xtrain1[i, j * 3] = data1[i, j]
        Xtrain1[i, j * 3 + 1] = data1[i][j + 1] + NUM_WORDS
        Xtrain1[i, j * 3 + 2] = data1[i][j + 2] + NUM_WORDS * 2
for i in range(25000):
    for j in range(MAX_LEN - 2):
        Xtest1[i, j * 3] = data2[i, j]
        Xtest1[i, j * 3 + 1] = data2[i][j + 1] + NUM_WORDS
        Xtest1[i, j * 3 + 2] = data2[i][j + 2] + NUM_WORDS * 2

Xtrain2 = np.zeros((25000, (MAX_LEN - 1) * 2), dtype=np.int)
Xtest2 = np.zeros((25000, (MAX_LEN - 1) * 2), dtype=np.int)
for i in range(25000):
    for j in range(MAX_LEN - 1):
        Xtrain2[i, j * 2] = data1[i, j]
        Xtrain2[i, j * 2 + 1] = data1[i][j + 1] + NUM_WORDS
for i in range(25000):
    for j in range(MAX_LEN - 1):
        Xtest2[i, j * 2] = data2[i, j]
        Xtest2[i, j * 2 + 1] = data2[i][j + 1] + NUM_WORDS

indice = np.arange(25000)
np.random.shuffle(indice)
Xtrain1 = Xtrain1[indice]
Xtrain2 = Xtrain2[indice]
Ytrain = Ytrain[indice]
Xtest1 = Xtest1[indice]
Xtest2 = Xtest2[indice]
Ytest = Ytest[indice]

#  build model
input1 = Input(shape=((MAX_LEN - 2) * 3,))
embedding1 = Embedding(NUM_WORDS * 3, 500, embeddings_initializer=keras.initializers.Orthogonal())(input1)
x = AveragePooling1D(pool_size=3)(embedding1)
x = GlobalMaxPooling1D()(x)

input2 = Input(shape=((MAX_LEN - 1) * 2,))
embedding2 = Embedding(NUM_WORDS * 2, 500, embeddings_initializer=keras.initializers.Orthogonal())(input2)
y = AveragePooling1D(pool_size=2, strides=2)(embedding2)
y = GlobalMaxPooling1D()(y)
z = Concatenate()([x, y])
# model.add(Dropout(0.5))
output = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit([Xtrain1, Xtrain2], Ytrain, batch_size=32, epochs=50, validation_data=([Xtest1, Xtest2], Ytest))

