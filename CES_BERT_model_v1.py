#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

!pip install tensorflow_text

import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

dat = pd.read_csv('/content/CESdat.csv')

dat.head(4)

# convert text to sequence of integers
max_words = 100
embed_dim = 128
lstm_out = 256
tokenizer = Tokenizer(num_words=max_words, split=' ')
texts = dat['CC15_300'].astype(str).values
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X)

# create target variable
y = pd.get_dummies(dat['ideology']).values
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")


bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(3, activation='softmax', name="output")(l)
# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

# convert text to sequence of integers
texts = dat['CC15_300'].astype(str).values
X = texts

# create target variable
y = pd.get_dummies(dat['ideology']).values

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Print model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size = 32)
#Accuracy: 46%.
