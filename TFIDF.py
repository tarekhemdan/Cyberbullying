#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Cyberbullying-Detection/Cyberbullying-Detection-on-Social-Media-using-Deep-Learning-and-Conventional-Machine-learning/blob/main/TFIDF.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import csv
import random
from collections import defaultdict, OrderedDict
from operator import add
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from tensorflow import keras
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


# In[ ]:


data_file = pd.read_csv('Final Dataset/cleaned_tweets_16K.csv')
data = data_file.drop(['contains_url','naughty_count','norm'],axis = 1)
data.head(10)


# # Get TF-IDF Data

# In[ ]:


def get_term_freq_data(use_idf):
    # Indicates if we are using TF or TF-IDF
    corpus = []
    y = []
    USE_IDF = use_idf
    dataset_filename='Final Dataset/cleaned_tweets_16K.csv'
    print("Using IDF: " + str(USE_IDF))

    # GET THE DATA
    #corpus, y = get_data(dataset_filename)
    with open(dataset_filename,newline='',encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        
        for row in csv_reader:
            if line_count == 0:
                print(','.join(row))
            else:
                corpus.append(row[1])
                y.append(int(row[0]))
            line_count += 1
            
    print("vectorising...")
    vec = TfidfVectorizer(min_df=0.001, max_df=1.0)
    
    # shuffle the data so that it is randomised
    X, Y = shuffle_data(corpus, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=14)  # 5 is best
    corpus_fit_transform = vec.fit_transform(corpus)

    newVec = TfidfVectorizer(vocabulary=vec.vocabulary_, use_idf=USE_IDF)
    X_train = newVec.fit_transform(X_train).toarray()
    X_test = newVec.fit_transform(X_test).toarray()
    print(X_train.shape)
    print(X_test.shape)
    print()

    return X_train, X_test, y_train, y_test


# # Logistic Regression

# In[ ]:


#Logistic Regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = get_term_freq_data(use_idf=True)
#repeat positives
# grams_X_train_logistic, grams_y_train_logistic = repeat_positives(X_train_logistic, y_train_logistic, repeats=2)

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


# # Random Forest

# In[ ]:


#Random Forest
X_train_random, X_test_random, y_train_random, y_test_random = get_term_freq_data(use_idf=True)
# repeat positives
# grams_X_train_random, grams_y_train_random = repeat_positives(X_train_random, y_train_random, repeats=2)
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


# # Bernoulli Naive Bayes

# In[ ]:


X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = get_term_freq_data(use_idf=True)
#repeat positives
# grams_X_train_bayes, grams_y_train_bayes = repeat_positives(X_train_bayes, y_train_bayes, repeats=2)
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


# # KNN

# In[ ]:


X_train_knn, X_test_knn, y_train_knn, y_test_knn = get_term_freq_data(use_idf=True)
# repeat positives
# grams_X_train_knn, grams_y_train_knn = repeat_positives(X_train_knn, y_train_knn, repeats=2)
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


# # AdaBoost Classifier

# In[ ]:


X_train_adaboost, X_test_adaboost, y_train_adaboost, y_test_adaboost = get_term_freq_data(use_idf=True)
#repeat positives
# grams_X_train_adaboost, grams_y_train_adaboost = repeat_positives(X_train_adaboost, y_train_adaboost, repeats=2)
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


# # SVM

# In[ ]:


X_train_svm, X_test_svm, y_train_svm, y_test_svm = get_term_freq_data(use_idf=True)
#repeat positives
#X_train_svm, y_train_svm = repeat_positives(X_train_svm, y_train_svm, repeats=2)
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


# # DECISION TREE

# In[ ]:


X_train_tree, X_test_tree, y_train_tree, y_test_tree = get_term_freq_data(use_idf=True)
#repeat positives
#grams_X_train_tree, grams_y_train_tree = repeat_positives(X_train_tree, y_train_tree, repeats=2)
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


import csv
from pathlib import Path

#Logistic
L_Recall = round(sm.recall_score(y_test_logistic, y_pred_logistic),4)
L_Precision = round(sm.precision_score(y_test_logistic, y_pred_logistic),4)
L_F1_score = round(sm.f1_score(y_test_logistic, y_pred_logistic), 4)
print("Logistic recall:",L_Recall )
print("Logistic precision:",L_Precision)
print("Logistic f1 score:",L_F1_score )

#Random
R_Recall = round(sm.recall_score(y_test_random, y_pred_random), 4)
R_Precision = round(sm.precision_score(y_test_random, y_pred_random), 4)
R_F1_score = round(sm.f1_score(y_test_random, y_pred_random), 4)
print("Random recall:",R_Recall )
print("Random precision:", R_Precision)
print("Random f1 score:",R_F1_score )

#Naive
N_Recall = round(sm.recall_score(y_test_bayes, y_pred_bayes), 4)
N_Precision =round(sm.precision_score(y_test_bayes, y_pred_bayes), 4)
N_F1_score = round(sm.f1_score(y_test_bayes, y_pred_bayes), 4)
print("Naive recall:", N_Recall)
print("Naive precision:",N_Precision )
print("Naive f1 score:", N_F1_score )

#KNN
KNN_Recall = round(sm.recall_score(y_test_knn, y_pred_knn), 4)
KNN_Precision =round(sm.precision_score(y_test_knn, y_pred_knn), 4)
KNN_F1_score = round(sm.f1_score(y_test_knn, y_pred_knn), 4)
print("KNN recall:", KNN_Recall )
print("KNN precision:", KNN_Precision)
print("KNN f1 score:", KNN_F1_score )

#Adaboost
Adaboost_Recall = round(sm.recall_score(y_test_adaboost, y_pred_adaboost), 4)
Adaboost_Precision = round(sm.precision_score(y_test_adaboost, y_pred_adaboost), 4)
Adaboost_F1_score = round(sm.f1_score(y_test_adaboost, y_pred_adaboost), 4)
print("KNN recall:",Adaboost_Recall )
print("KNN precision:",Adaboost_Precision )
print("KNN f1 score:", Adaboost_F1_score)

#SVM
svm_Recall = round(sm.recall_score(y_test_svm, y_pred_svm), 4)
svm_Precision = round(sm.precision_score(y_test_svm, y_pred_svm), 4)
svm_F1_score = round(sm.f1_score(y_test_svm, y_pred_svm), 4)
print("SVM recall:",svm_Recall )
print("SVM precision:", svm_Precision )
print("SVM f1 score:", svm_F1_score )

#Decision
decision_Recall = round(sm.recall_score(y_test_tree, y_pred_tree), 4)
decision_Precision = round(sm.precision_score(y_test_tree, y_pred_tree), 4)
decision_F1_score = round(sm.f1_score(y_test_tree, y_pred_tree), 4)
print("Decision recall:", decision_Recall)
print("Decision precision:",decision_Precision )
print("Decision f1 score:", decision_F1_score )


# final_data = [L_Recall,L_Precision, L_F1 score, R_Recall, 
#               R_Precision,R_F1 score,N_Recall,N_Precision, N_F1 score,
#               Adaboost_Recall,Adaboost_Precision,Adaboost_F1 score,svm_Recall,
#              svm_Precision,svm_F1 score,decision_Recall,decision_Precision,decision_F1 score]


with open('TFIDF_Final_data5.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['Recall','Precision','F1-score'])
    writer.writerow([L_Recall,L_Precision,L_F1_score])
    writer.writerow([R_Recall, R_Precision,R_F1_score])
    writer.writerow([N_Recall,N_Precision, N_F1_score])
    writer.writerow([KNN_Recall,KNN_Precision,KNN_F1_score])
    writer.writerow([Adaboost_Recall,Adaboost_Precision,Adaboost_F1_score])
    writer.writerow([svm_Recall,svm_Precision,svm_F1_score])
    writer.writerow([decision_Recall,decision_Precision,decision_F1_score])
    
    

