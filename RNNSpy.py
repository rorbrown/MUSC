from load_cr import loadReports
import tensorflow as tf
from keras.layers import Dense, Flatten, Reshape, BatchNormalization, CuDNNLSTM, CuDNNGRU
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import load_model
import numpy as np
import random
import matplotlib.pyplot as plt


#so it doesn't use ellipse when printing
np.set_printoptions(threshold=np.inf)

#Loading data and padding it
Data = loadReports()
txt = Data["txt"]
labels = np.array(Data["label"])

vocab_size = 10000
encoded_docs = [one_hot(t, vocab_size) for t in txt]
max_length = 300
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


print("X:", padded_docs.shape)
print("Labels:", labels.shape)


#RANDOMIZE:
randArray = list(zip(padded_docs, labels))
random.shuffle(randArray)

X, labels = zip(*randArray)
X = np.array(X)
labels = np.array(labels)

# Epsilon val_acc Results:
#   2 layers 64, 32 and 32 embedding
#   1 --> 64.18%
#   .7 --> 84.02
#   .4 --> 93.04
#   .1 --> 91.24%
#   .01 --> 83.25%
#   .001 --> 84.54% *increased 30% on last 3 epochs
#   .0001 --> 87.11 *fluctuated a lot
#   0 --> 76.29% *may need more epochs

#   2 layers 32, 32
#   .7 --> 53.09%
#   .4 --> 89.43%
#   .1 --> 83.25%

#   2 layers 64,32 with 100 on embedding
#   .4 --> 90.98%
#   .1 --> 50.93%

#   3 layers 100,64,32
#   .8 --> 86.86%
#   .7 --> 87.89%
#   .6 --> 89.18%
#   .4 --> 75.26%
#   3 layers 100,100,100
#   .7 --> 84.02%
#   .8 --> 88.92%
#    .9 --> 88.14%
#    1 --> 75%

#   4 layers 100, 64,32 and dense 10
#   .8 --> 84.02%
#   .7 --> 78.09%

#   2 layers GRU 64,64
#   .3 --> 91.24%
#   .4 --> 83.76%
#   .5 --> 88.92
#   .6 --> 89.18
epsilonVals = [.65]

for epsilonVal in epsilonVals:
    print("Epsilon:", epsilonVal)
    #Define the RNN model
    rnn = Sequential()
    rnn.add(Embedding(vocab_size, 200, input_length=max_length))
    rnn.add(CuDNNGRU(128, return_sequences=True))
    rnn.add(CuDNNGRU(64, return_sequences=True))
    rnn.add(CuDNNGRU(32))
    #rnn.add(CuDNNLSTM(100, return_sequences = True))
    #rnn.add(CuDNNLSTM(64, return_sequences = True))
    #rnn.add(CuDNNLSTM(32))
    rnn.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=epsilonVal))
    rnn.add(Dense(32, activation='relu'))
    rnn.add(Dense(1, activation='sigmoid'))
    rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(rnn.summary())
    history = rnn.fit(X, labels, epochs = 40, validation_split=.3)
    #Evaluate on Training and validation data
    loss, accuracy = rnn.evaluate(X, labels, verbose=0)
    print('Accuracy: %f' %(accuracy*100))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




##Save Rnn to a file
# rnn.save("rnnModel.h5")
# print("Saved model to disk")
