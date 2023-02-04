#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


import tensorflow as tf


# In[4]:


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


# In[5]:


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


# In[6]:


dat = pd.read_csv('CCES_dat.csv')
dat


# In[7]:


import string

# Define a translation table that will remove punctuation
trans_table = str.maketrans('', '', string.punctuation)

# Use the translate method to remove punctuation from each string in the 'CC15_300' column
dat['CC15_300'] = dat['CC15_300'].astype(str).apply(lambda x: x.translate(trans_table))
dat['CC15_300']


# In[8]:


dat = dat[dat['CC15_300'].astype(str).ne("")]
dat.dropna(subset=['CC15_300'], inplace=True)
dat


# In[9]:


dat['ideology'] = np.nan
dat.loc[dat['ideo5'].isin(["Very liberal", "Liberal"]), 'ideology'] = 0
dat.loc[dat['ideo5'] == "Moderate", 'ideology'] = 1
dat.loc[dat['ideo5'].isin(["Conservative", "Very conservative"]), 'ideology'] = 2
dat.dropna(subset=['ideology'], inplace=True)


# In[10]:


#Convert target to categorcical var
dat['ideology'] = dat['ideology'].astype('category')


# In[11]:


# convert text to sequence of integers
max_words = 500
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(dat['CC15_300'].values)
X = tokenizer.texts_to_sequences(dat['CC15_300'].values)
X = pad_sequences(X)


# In[12]:


# create target variable
y = pd.get_dummies(dat['ideology']).values


# In[13]:


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[14]:


# create the model
embed_dim = 128
lstm_out = 196


# In[15]:


model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[16]:


# train the model
batch_size = 32
epochs = 8
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)


# In[17]:


# evaluate the model
score, acc = model.evaluate(X_test, y_test, verbose=2, batch_size=batch_size)
print("Test accuracy:", acc)

