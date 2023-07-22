#!/usr/bin/env python
# coding: utf-8

# In[202]:


# # import libraries 
import numpy as np
import pandas as pd
from pandas import DataFrame
import itertools
import csv,codecs,nltk,re
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
#from sklearn import cross_validation#    from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, NuSVC , LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Activation


# # Read the dataset

# In[203]:


data = pd.read_csv('/kaggle/input/cyberbullying-dataset/twitter_racism_parsed_dataset.csv')

texts = data['Text'].astype(str)
labels = data['oh_label'].astype(int)


# # text preprocessing

# In[204]:


def stopwordremoval(Text):
    stop=stopwords.words('english')
    needed_words=[]
    words=word_tokenize(Text)
    for w in words:
         if len(w)>=2 and w not in stop:
                needed_words.append(w)
    filterd_sent= " ".join(needed_words)
    return filterd_sent
def stemming(Text):
    st = ISRIStemmer()
    stemmed_words=[]
    words=word_tokenize(Text)
    for w in words:
        stemmed_words.append(st.stem(w))
    stemmed_sent=" ".join(stemmed_words)
    return stemmed_sent
def preparedatasets(data):
    sentences=[]
    for index,r in data.iterrows():
        Text=stopwordremoval(r['Text'])
        Text=stemming(r['Text'])
        sentences.append([r['Text'],r['oh_label']])
    df_sentence=DataFrame(sentences, columns=["Text", "oh_label"])
    return df_sentence  


# In[205]:


data=preparedatasets(data)


# In[206]:


#data=preparedatasets(data)
data.dropna(inplace=True)
data.head()
data['word_count'] = data['Text'].apply(lambda x: len(str(x).split()))
#Remove 0 and 1 word_count posts
new_data=data[(data.word_count >1)]
new_data.describe()    
data=new_data


# In[207]:


data=data.drop_duplicates( keep="first")


# In[208]:


# Preprocess the data
texts = data['Text'].astype(str)
labels = data['oh_label'].astype(int)


# In[209]:


#X_train, X_test, y_train, y_test = \
#train_test_split(data['Text'], data['oh_label'],test_size=0.20, random_state = 0)
#vectorizer = TfidfVectorizer( analyzer='word',smooth_idf=True, ngram_range=(1,2))
#vectorizer.fit(X_train)
#X_train_vectorized = vectorizer.transform(X_train)

   # Tokenize the text data
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
text_data = keras.preprocessing.sequence.pad_sequences(text_sequences)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2)


# In[210]:


# ---------- ML Algorithms-----------
#cl=MultinomialNB()
#cl=LogisticRegression()
#cl=SVC()
#cl=DecisionTreeClassifier()
#cl=RandomForestClassifier()


# In[211]:


# ---------- Deep Learning Algorithms ----------

# ------------  LSTM --------------
model = keras.Sequential()
model.add(keras.layers.Embedding(5000, 32, input_length=text_data.shape[1]))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(1, activation='sigmoid'))


# In[212]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[213]:


#Number of words in the dataset
def get_count(data):
    d2=[]
    tt=""
    for index,r in data.iterrows():
        d2.append((r['Text'] ))
        tt+=r['Text']
    return len(tt)  


# In[214]:


#cl.fit(X_train_vectorized, y_train)

history=model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[216]:


# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred 


# In[218]:


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# In[219]:


# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:', cm)


# In[ ]:




