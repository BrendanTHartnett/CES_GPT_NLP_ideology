#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os

import matplotlib
import pandas as pd
import glob
import sklearn.metrics
import re
import numpy as np
import seaborn as sns
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
sns.set() # use seaborn plotting style
from sklearn.utils import resample
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from numpy import ravel
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from datetime import date
import snscrape.modules.twitter as sntwitter
import glob
import re
import time
import schedule
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[1]:


dat = pd.read_csv('CCES_dat.csv')
dat


# In[17]:


import string

# Define a translation table that will remove punctuation
trans_table = str.maketrans('', '', string.punctuation)

# Use the translate method to remove punctuation from each string in the 'CC15_300' column
dat['CC15_300'] = dat['CC15_300'].astype(str).apply(lambda x: x.translate(trans_table))
dat['CC15_300']


# In[18]:


dat = dat[dat['CC15_300'].astype(str).ne("")]
dat.dropna(subset=['CC15_300'], inplace=True)
dat


# In[19]:


stemmer=PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


# In[36]:


dat['stemmed_text'] = dat['CC15_300'].apply(stem_sentences)


# In[37]:


dat['ideology'] = np.nan
dat.loc[dat['ideo5'].isin(["Very liberal", "Liberal"]), 'ideology'] = 0
dat.loc[dat['ideo5'] == "Moderate", 'ideology'] = 1
dat.loc[dat['ideo5'].isin(["Conservative", "Very conservative"]), 'ideology'] = 2
dat.dropna(subset=['ideology'], inplace=True)


# In[38]:


#Save data before splitting
saved_data = dat


# In[39]:


train_data, test_data = sklearn.model_selection.train_test_split(saved_data, train_size = 0.7, random_state=143)


# In[40]:


model = make_pipeline(TfidfVectorizer(max_features=400, ngram_range=(1, 3), stop_words='english'),
                      SelectKBest(chi2, k=100),
                      BernoulliNB())
model.fit(train_data.stemmed_text, train_data.ideology)


# In[41]:


test_data['predicted_class'] = 0 
test_data['predicted_class'] = model.predict(test_data.stemmed_text)
accuracy = accuracy_score(test_data.predicted_class, test_data.ideology)
accuracy


# In[42]:


mat = confusion_matrix(test_data.ideology, test_data.predicted_class, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
plt.show()

