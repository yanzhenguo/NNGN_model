# coding=utf-8
import numpy as np
import keras
from keras.layers import Dense, GlobalMaxPooling1D, Input, Embedding, \
    AveragePooling1D, GlobalAveragePooling1D, Activation, Conv1D, Dropout, MaxPooling1D, LSTM, Flatten, Concatenate
from keras.models import Model, load_model, Sequential
import data_process
"implementation of bow-NNGN"

NUM_WORDS = 30000
MAX_LEN = 300

imdb = data_process.ImdbCorpus(num_words=NUM_WORDS,max_len=MAX_LEN)
Xtrain, Ytrain, Xtest, Ytest = imdb.get_input()
# shuffle the data
indice = np.arange(len(Xtrain))
np.random.shuffle(indice)
Xtrain = Xtrain[indice]
Xtest = Xtest[indice]
Ytrain = Ytrain[indice]
Ytest = Ytest[indice]
# build model
main_input = Input(shape=(MAX_LEN,))
init_method = keras.initializers.Orthogonal()
x = Embedding(NUM_WORDS, 1000)(main_input)
x = AveragePooling1D(pool_size=3, strides=1)(x)
x = GlobalMaxPooling1D()(x)
output = Dense(1, activation='sigmoid', trainable=True, use_bias=True)(x)
model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
# train on the data
model.fit(Xtrain, Ytrain, batch_size=32, epochs=20, validation_data=(Xtest, Ytest))