from load_cr import loadReports
import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Embedding, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import numpy as np
import random
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score


#####Loading data and padding it#####
def load_data():
    Data = loadReports()
    txt = Data["txt"]
    labels = np.array(Data["label"])
    vocab_size = 10000
    encoded_docs = [one_hot(t, vocab_size) for t in txt]
    max_length = 600
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("X:", padded_docs.shape)
    print("Labels:", labels.shape)

    #RANDOMIZE:
    randArray = list(zip(padded_docs, labels))
    #random.shuffle(randArray)

    X, labels = zip(*randArray)
    X = np.array(X)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, shuffle= False)


    return x_train, x_test, y_train, y_test



#####Loading Glove Embeddings#####
def load_Glove(vocab_size):
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



def build_cnn():

    # model variables
    hidden_dims = 128
    filt_sz = [3, 4, 5]
    drop_rate = 0.2
    EMBEDDING_DIM = 300
    num_filters = 100
    maxlen = 600
    vocab_size = 10000
    pre_trained= False
    lr = .0004



    if pre_trained == False:
        embedding_lyr = Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length = maxlen)
    else:
        embedding_matrix = load_Glove(vocab_size)
        vocab_size = embedding_matrix.shape[0]
        embedding_lyr = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)


    model = Sequential()
    model.add(embedding_lyr)
    model.add(Dropout(drop_rate))
    model.add(Conv1D(num_filters, filt_sz[0], activation = "relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(num_filters, filt_sz[1], activation = "relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(num_filters, filt_sz[2], activation = "relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hidden_dims,activation = "relu"))
    model.add(Dropout(drop_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr = lr)
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
    print(model.summary())
    return model, lr


def run():
    #Get Data
    x_train, x_test, y_train, y_test = load_data()

    model, lr = build_cnn()
    epochs = int(.015/lr)
    history = model.fit(x_train, y_train, epochs = epochs, validation_split=.1)


    ##Evaluation of the model##
    test = model.evaluate(x_test, y_test)
    print("Test Data Performance [loss, acc]:", test)
    y_pred = model.predict(x_test)
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


done = run()
