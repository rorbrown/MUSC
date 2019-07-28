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
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

def load_data(vocab_size, max_length):
	#so it doesn't use ellipse when printing
	np.set_printoptions(threshold=np.inf)

	#Loading data and padding it
	Data = loadReports()
	txt = Data["txt"]
	labels = np.array(Data["label"])


	encoded_docs = [one_hot(t, vocab_size) for t in txt]

	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	print("X:", padded_docs.shape)
	print("Labels:", labels.shape)


	#RANDOMIZE:
	randArray = list(zip(padded_docs, labels))
	#random.shuffle(randArray)

	X, labels = zip(*randArray)
	X = np.array(X)
	labels = np.array(labels)
	x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle= False)


	return x_train, x_test, y_train, y_test





# load the whole embedding into memory
#****need to make this a function that only runs once*******
def load_Glove():
	Data = loadReports()
	txt = Data["txt"]

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


	return embedding_matrix





#Creating the model
#Best Results:
#	.9,.95,1 without weight regularization --> 93-96%
#	1 with weight regularization --> max of 94.75%

def model(pre_trained, vocab_size, max_length, x_train, x_test, y_train, y_test):

	epsilonVals = [.95]

	for epsilonVal in epsilonVals:

		print("Epsilon:", epsilonVal)

		logdir="logs/scalars/bi/42b/119/" +str(epsilonVal)
		tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
		#Define the RNN model
		rnn = Sequential()
		if pre_trained == True:
			embedding_matrix = load_Glove()
			e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
		else:
			e = Embedding(vocab_size, 300, input_length=max_length)

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
		optimizer = Adam(lr = .001, decay = .00005)
		rnn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		print(rnn.summary())
		history = rnn.fit(x_train, y_train, epochs = 20, validation_split=.01, callbacks=[tensorboard_callback])

		##Evaluation of the model##
		test = rnn.evaluate(x_test, y_test)
		print("Test Data Performance:", test)
		y_pred = rnn.predict(x_test)
		auc = roc_auc_score(y_test, y_pred)
		print("AUC:", auc)

		predictions = np.zeros(y_pred.shape[0])
		j = 0
		for i in y_pred:
			if i > .5:
				predictions[j] = 1
			else:
				predictions[j] = 0
			j += 1
		f1 = f1_score(y_test, predictions)
		print("F1 score:", f1)


def run():
	vocab_size = 14270 #size of embedding
	max_length = 600
	pre_trained = True


	x_train, x_test, y_train, y_test = load_data(vocab_size, max_length)

	rnn = model(pre_trained, vocab_size, max_length, x_train, x_test, y_train, y_test)



done = run()


##Save Rnn to a file
# rnn.save("biRnnGloveModel.h5")
# print("Saved model to disk")
