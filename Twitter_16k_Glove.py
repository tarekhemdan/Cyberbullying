#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Cyberbullying-Detection/Cyberbullying-Detection-on-Social-Media-using-Deep-Learning-and-Conventional-Machine-learning/blob/main/Twitter_16k_Glove.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import csv
import random
from collections import defaultdict, OrderedDict
from operator import add
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn
import sklearn.metrics as sm
from sklearn import svm, tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# In[ ]:


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


# # Get Glove Data

# In[ ]:


def get_glove_data():
    comments = []
    y = []
    dataset_filename='Final Dataset/cleaned_tweets_16K.csv'
    
    with open(dataset_filename,newline='',encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        
        for row in csv_reader:
            if line_count == 0:
                print(','.join(row))
            else:
                comments.append(row[1])
                y.append(int(row[0]))
            line_count += 1
    
    num_comments = len(comments)
    print("splitting data........")
    word_arrays = []
    for s in comments:
        word_arrays.append(s.split(' '))
        
    print("Getting GLOVE embeddings size 50..")
    file = open('glove.6B/glove.6B.50d.txt',errors='ignore').readlines()
    gloveDict = {}
    for line in file:
        info = line.split(' ')
        key = info[0]
        vec = []
        for elem in info[1:]:
            vec.append(elem.rstrip())
        gloveDict[key] = vec
    print(len(gloveDict),"words in the GLOVE dictionary\n")
    
    #VECTORISE WORDS
    print("converting comments to lists of vectors........")
    word_vectors = []
    for sentence in word_arrays:
        temp = []
        for word in sentence:
            if word in gloveDict:
                temp.append(gloveDict[word])
        word_vectors.append(temp)
        
    MAX_LEN = 32
    
    print("padding vectors to maxlen = ",MAX_LEN,".....")
    padded_word_vecs = np.array(pad_sequences(word_vectors, padding='pre', maxlen=MAX_LEN, dtype='float32'))
    padded_word_vecs = padded_word_vecs.reshape((num_comments, -1))
    
    print("DONE PRE-PROCESSING\n")
    
    #CLASSIFYING
    print("splitting")
    X_train,X_test,y_train,y_test = train_test_split(padded_word_vecs,y,test_size=0.20)
    
    return X_train, X_test, y_train, y_test


# # Logistic Regression

# In[ ]:


#Logistic Regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = get_glove_data()

grid_searching = False
clf = sklearn.linear_model.LogisticRegression(penalty="l2", max_iter=100, solver="liblinear")
clf = clf.fit(X_train_logistic, y_train_logistic)

#PREDICT
print("\nevaluating")
y_pred_logistic = clf.predict(X_test_logistic)
print(y_pred_logistic)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_logistic,y_pred_logistic))
print("accuracy:", round(sm.accuracy_score(y_test_logistic,y_pred_logistic), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_random, y_pred_random), 4))
print("precision:", round(sm.precision_score(y_test_random, y_pred_random), 4))
print("f1 score:", round(sm.f1_score(y_test_random, y_pred_random), 4))


# # Random Forest

# In[ ]:


#Logistic Regression
X_train_random, X_test_random, y_train_random, y_test_random = get_glove_data()
grid_searching = False
clf = RandomForestClassifier(n_estimators=100, max_depth=4)
clf = clf.fit(X_train_random, y_train_random)

#PREDICT
print("\nevaluating")
y_pred_random = clf.predict(X_test_random)
print(y_pred_random)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_random,y_pred_random))
print("accuracy:", round(sm.accuracy_score(y_test_random,y_pred_random), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_random, y_pred_random), 4))
print("precision:", round(sm.precision_score(y_test_random, y_pred_random), 4))
print("f1 score:", round(sm.f1_score(y_test_random, y_pred_random), 4))


# # Bernoulli Naive Bayes

# In[ ]:


X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = get_glove_data()
grid_searching = False
clf = BernoulliNB()
clf = clf.fit(X_train_bayes, y_train_bayes)

#PREDICT
print("\nevaluating")
y_pred_bayes = clf.predict(X_test_bayes)
print(y_pred_bayes)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_bayes,y_pred_bayes))
print("accuracy:", round(sm.accuracy_score(y_test_bayes,y_pred_bayes), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_bayes, y_pred_bayes), 4))
print("precision:", round(sm.precision_score(y_test_bayes, y_pred_bayes), 4))
print("f1 score:", round(sm.f1_score(y_test_bayes, y_pred_bayes), 4))


# # KNN

# In[ ]:


X_train_knn, X_test_knn, y_train_knn, y_test_knn = get_glove_data()
grid_searching = False
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X_train_knn, y_train_knn)

#PREDICT
print("\nevaluating")
y_pred_knn = clf.predict(X_test_knn)
print(y_pred_knn)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_knn,y_pred_knn))
print("accuracy:", round(sm.accuracy_score(y_test_knn,y_pred_knn), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_knn, y_pred_knn), 4))
print("precision:", round(sm.precision_score(y_test_knn, y_pred_knn), 4))
print("f1 score:", round(sm.f1_score(y_test_knn, y_pred_knn), 4))


# # Adaboost Classifier

# In[ ]:


X_train_adaboost, X_test_adaboost, y_train_adaboost, y_test_adaboost = get_glove_data()
grid_searching = False
clf = AdaBoostClassifier()
clf = clf.fit(X_train_adaboost, y_train_adaboost)

#PREDICT
print("\nevaluating")
y_pred_adaboost = clf.predict(X_test_adaboost)
print(y_pred_adaboost)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_adaboost,y_pred_adaboost))
print("accuracy:", round(sm.accuracy_score(y_test_adaboost,y_pred_adaboost), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_adaboost, y_pred_adaboost), 4))
print("precision:", round(sm.precision_score(y_test_adaboost, y_pred_adaboost), 4))
print("f1 score:", round(sm.f1_score(y_test_adaboost, y_pred_adaboost), 4))


# # SVM

# In[ ]:


X_train_svm, X_test_svm, y_train_svm, y_test_svm = get_glove_data()
grid_searching = False
clf = svm.SVC(C=10, kernel="rbf", gamma=0.001)
clf = clf.fit(X_train_svm, y_train_svm)

#PREDICT
print("\nevaluating")
y_pred_svm = clf.predict(X_test_svm)
print(y_pred_svm)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_svm,y_pred_svm))
print("accuracy:", round(sm.accuracy_score(y_test_svm,y_pred_svm), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_svm, y_pred_svm), 4))
print("precision:", round(sm.precision_score(y_test_svm, y_pred_svm), 4))
print("f1 score:", round(sm.f1_score(y_test_svm, y_pred_svm), 4))


# # Decision Tree

# In[ ]:


X_train_tree, X_test_tree, y_train_tree, y_test_tree = get_glove_data()
grid_searching = False
clf = clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_tree, y_train_tree)

#PREDICT
print("\nevaluating")
y_pred_tree = clf.predict(X_test_tree)
print(y_pred_tree)


# In[ ]:


# EVALUATE
print("confusion matrix:\n", sm.confusion_matrix(y_test_tree,y_pred_tree))
print("accuracy:", round(sm.accuracy_score(y_test_tree,y_pred_tree), 4))


# In[ ]:


print("recall:", round(sm.recall_score(y_test_tree, y_pred_tree), 4))
print("precision:", round(sm.precision_score(y_test_tree, y_pred_tree), 4))
print("f1 score:", round(sm.f1_score(y_test_tree, y_pred_tree), 4))

