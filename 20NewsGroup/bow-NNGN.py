import codecs
import pickle
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, Embedding, \
    GlobalAveragePooling1D,MaxPooling2D,AveragePooling1D
import data_process

num_words=30000
max_len=1000
num_train=11314
num_test=7532

#prepare the data
texts, Ytrain, Ytest = data_process.extract_data()
Ytrain=to_categorical(Ytrain)
Ytest=to_categorical(Ytest)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts[:num_train])

sequence1 = tokenizer.texts_to_sequences(texts[:num_train])
sequence2 = tokenizer.texts_to_sequences(texts[num_train:])
word_index = tokenizer.word_index

# sequences=[]
# for i in range(50000):
#     t=[]
#     tokens=texts[i].lower().split(' ')
#     for j in range(len(tokens)):
#         index=word_index.get(tokens[j],0)
#         if index<num_words:
#             t.append(index)
#         else:
#             t.append(0)
#     sequences.append(t)
data1 = pad_sequences(sequence1, maxlen=max_len)
data2 = pad_sequences(sequence2, maxlen=max_len)
# shuffle the data
indice1 = np.arange(num_train)
np.random.shuffle(indice1)
Xtrain = data1[indice1]
Ytrain = Ytrain[indice1]
indice2 = np.arange(num_test)
np.random.shuffle(indice2)
Xtest = data2[indice2]
Ytest = Ytest[indice2]
# build the model
input = Input(shape=(max_len,))
embedding1 = Embedding(num_words, 1000, embeddings_initializer=keras.initializers.Orthogonal())(input)
x = AveragePooling1D(pool_size=3, strides=1)(embedding1)
x = GlobalMaxPooling1D()(x)
output = Dense(20, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
# train on the data
model.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data=(Xtest, Ytest))
