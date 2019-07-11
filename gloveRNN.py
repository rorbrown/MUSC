from load_cr import loadReports
import tensorflow as tf
from keras.layers import Dense, BatchNormalization, CuDNNLSTM, CuDNNGRU, Bidirectional, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt
import keras


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
#   1.5 --> 92.78


# load the whole embedding into memory
#****need to make this a function that only runs once*******
embeddings_index = dict()
f = open('./glove.42B.300d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
t = Tokenizer()
t.fit_on_texts(txt)
vocab_size = len(t.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

#Creating the model

epsilonVals = [.9,.95,1]

for epsilonVal in epsilonVals:

    print("Epsilon:", epsilonVal)

    logdir="logs/scalars/bi/42b/1/" +str(epsilonVal)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    #Define the RNN model
    rnn = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
    rnn.add(e)
    rnn.add(Bidirectional(CuDNNGRU(256, return_sequences=True)))
    rnn.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
    rnn.add(Bidirectional(CuDNNGRU(64, return_sequences=True)))
    rnn.add(Bidirectional(CuDNNGRU(32)))
    #rnn.add(CuDNNLSTM(100, return_sequences = True))
    #rnn.add(CuDNNLSTM(64, return_sequences = True))
    #rnn.add(CuDNNLSTM(32))
    rnn.add(BatchNormalization(axis=-1, momentum=.99, epsilon=epsilonVal))
    rnn.add(Dense(32, activation='relu'))
    rnn.add(Dense(1, activation='sigmoid'))
    #optimizer = Adam(lr = .001, decay = .00005)
    rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(rnn.summary())
    history = rnn.fit(X, labels, epochs = 20, validation_split=.25, callbacks=[tensorboard_callback])
    #Evaluate on Training data
    loss, accuracy = rnn.evaluate(X, labels, verbose=0)
    print('Accuracy: %f' %(accuracy*100))
	#
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()




##Save Rnn to a file
# rnn.save("biRnnGloveModel.h5")
# print("Saved model to disk")
