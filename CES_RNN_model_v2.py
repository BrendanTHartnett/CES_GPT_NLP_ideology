#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.utils import resample
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from numpy import ravel
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import gensim.downloader as api

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Dropout


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

#Import CES data
dat = pd.read_csv('CCES_dat.csv')

import string
trans_table = str.maketrans('', '', string.punctuation)

dat['CC15_300'] = dat['CC15_300'].astype(str).apply(lambda x: x.translate(trans_table))
dat = dat[dat['CC15_300'].astype(str).ne("")]
dat.dropna(subset=['CC15_300'], inplace=True)
dat['ideology'] = np.nan
dat.loc[dat['ideo5'].isin(["Very liberal", "Liberal"]), 'ideology'] = 0
dat.loc[dat['ideo5'] == "Moderate", 'ideology'] = 1
dat.loc[dat['ideo5'].isin(["Conservative", "Very conservative"]), 'ideology'] = 2
dat.dropna(subset=['ideology'], inplace=True)
#Convert target to categorcical var
dat['ideology'] = dat['ideology'].astype('category')

# convert text to sequence of integers
max_words = 100
embed_dim = 128
lstm_out = 256
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(dat['CC15_300'].values)
X = tokenizer.texts_to_sequences(dat['CC15_300'].values)
X = pad_sequences(X)

#import embeddings from webpage listed
embeddings_index = np.load("https://huggingface.co/fse/fasttext-wiki-news-subwords-300/blob/main/fasttext-wiki-news-subwords-300.model.vectors.npy", allow_pickle=True)

# load pre-trained embeddings
embed_dim = 300
embedding_matrix = np.zeros((max_words, embed_dim))
for word, i in tokenizer.word_index.items():
    if i >= max_words:
        continue
    try:
        embedding_vector = embeddings_index[tokenizer.word_index[word]]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass

tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(dat['CC15_300'].values)
X = tokenizer.texts_to_sequences(dat['CC15_300'].values)
X = pad_sequences(X)
# create target variable
y = pd.get_dummies(dat['ideology']).values
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# create the model
embed_dim = 300
lstm_out = 256
model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(GRU(lstm_out, return_sequences=True)))
model.add(Bidirectional(GRU(lstm_out, return_sequences=False)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# set early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# train the model
batch_size = 32
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# evaluate the model
score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
print("Test accuracy:", acc)

#Plot model performance
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
