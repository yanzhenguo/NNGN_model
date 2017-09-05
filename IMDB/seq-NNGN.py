# coding=utf-8
import numpy as np
import keras
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Activation, Conv1D, Dropout, MaxPooling1D, LSTM, Flatten, Concatenate
from keras.models import Model, load_model, Sequential
import data_process
"implementation of seq-NNGN with region size to be 3"

NUM_WORDS = 30000
MAX_LEN = 300

imdb = data_process.ImdbCorpus(num_words=NUM_WORDS, max_len=MAX_LEN)
data1, Ytrain, data2, Ytest = imdb.get_input()
Xtrain = np.zeros((25000, (MAX_LEN - 2) * 3), dtype=np.int)
Xtest = np.zeros((25000, (MAX_LEN - 2) * 3), dtype=np.int)
for i in range(25000):
    for j in range(MAX_LEN - 2):
        Xtrain[i, j * 3] = data1[i, j]
        Xtrain[i, j * 3 + 1] = data1[i][j + 1] + NUM_WORDS
        Xtrain[i, j * 3 + 2] = data1[i][j + 2] + NUM_WORDS * 2
for i in range(25000):
    for j in range(MAX_LEN - 2):
        Xtest[i, j * 3] = data2[i, j]
        Xtest[i, j * 3 + 1] = data2[i][j + 1] + NUM_WORDS
        Xtest[i, j * 3 + 2] = data2[i][j + 2] + NUM_WORDS * 2
# shuffle the data
indice = np.arange(len(Xtrain))
np.random.shuffle(indice)
Xtrain = Xtrain[indice]
Xtest = Xtest[indice]
Ytrain = Ytrain[indice]
Ytest = Ytest[indice]
# build model
main_input = Input(shape=((MAX_LEN - 2) * 3,))
embedding1 = Embedding(NUM_WORDS * 3, 500, embeddings_initializer=keras.initializers.Orthogonal())(main_input)
# embedding1 = Embedding(num_words * 3, 500)(main_input)
x = AveragePooling1D(pool_size=3)(embedding1)
x = GlobalMaxPooling1D()(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
# train on the data
model.fit(Xtrain, Ytrain, batch_size=32, epochs=20, validation_data=(Xtest, Ytest))
