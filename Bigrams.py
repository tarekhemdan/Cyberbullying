#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Cyberbullying-Detection/Cyberbullying-Detection-on-Social-Media-using-Deep-Learning-and-Conventional-Machine-learning/blob/main/Bigrams.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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


file = pd.read_csv('Final Dataset/cleaned_tweets_16K.csv')
data = file.drop(['contains_url','naughty_count','norm'],axis = 1)
data.head(10)


# In[ ]:


print(file["text_message"].describe())


# In[ ]:


file["text_message"].value_counts()


# In[ ]:


data.value_counts


# In[ ]:


pd.unique(data["label_bullying"])


# In[ ]:


missing = data.isnull()
missing.sorted(ascending=False)


# In[ ]:


missing = (data.isnull().sum()/len(data)*100).round(2)
pd.DataFrame({'% missing values': missing})


# # Class Distribution

# In[ ]:


ax = sns.countplot(x = 'label_bullying', data = data)
ax.set(xlabel='Class', ylabel='Count', title = 'Class Distribution')
plt.show()


# In[ ]:


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


# # Get Bigrams_Data

# In[ ]:


def generate_all_char_bigrams():
    bigram_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(0, 26):
        for j in range(0, 26):
            gram = str(alphabet[i])+str(alphabet[j])
            bigram_dict[gram] = 0

    return OrderedDict(sorted(bigram_dict.items()))


# In[ ]:


def generate_all_char_trigrams():
    bigram_dict = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(0, 26):
        for j in range(0, 26):
            for k in range(0, 26):
                gram = str(alphabet[i])+str(alphabet[j])+str(alphabet[k])
                bigram_dict[gram] = 0
    return OrderedDict(sorted(bigram_dict.items()))


# In[ ]:


def get_ngram_data(ngram_size, N=20000):
    dataset_filename='Final Dataset/cleaned_tweets_16K.csv'
    X = []
    y = []
    print("\nGETTING DATA - " + str(ngram_size) + "-grams")

    # compute all possible n-grams and create a base dictionary for counting them
    if ngram_size == 2:
        global_grams = generate_all_char_bigrams()
    else:
        global_grams = generate_all_char_trigrams()
        

    # READ CSV
    with open(dataset_filename,newline='',encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(','.join(row))
            else:
                if line_count < N:
                    if line_count % 200 == 0:
                        print(str(line_count) + " ngrams computed")

                    label_bullying = int(row[0])
                    text_message = row[1]

                    # current features
                    temp_x = []
                    this_bigram_dict = global_grams.copy()

                    # split text messages into a list of its ngrams
                    ngram = [text_message[j:j+ngram_size] for j in range(len(text_message)-(ngram_size-1))]

                    # TODO: change this to 'count' so we get better performance
                    # count occurences of each character ngram
                    for gram in ngram:
                        if gram in this_bigram_dict:
                            this_bigram_dict[gram] += 1

                    # create feature vector for this instance (take just the values)
                    for key in this_bigram_dict:
                        temp_x.append(this_bigram_dict[key])

                    X.append(temp_x)
                    y.append(label_bullying)

                    del this_bigram_dict
            line_count += 1
    print("processed", line_count-1, "comments\n")

    # shuffle the data so that it is randomised
    X, y = shuffle_data(X, y)

    # SPLIT
    print("splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

    return X_train, X_test, y_train, y_test


# In[ ]:





# # Logistic Regression

# In[ ]:


#Logistic Regression
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = get_ngram_data(ngram_size=2, N=69446)

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
X_train_random, X_test_random, y_train_random, y_test_random = get_ngram_data(ngram_size=2, N=69446)
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


X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = get_ngram_data(ngram_size=2, N=69446)
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


X_train_knn, X_test_knn, y_train_knn, y_test_knn = get_ngram_data(ngram_size=2, N=69446)
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


X_train_adaboost, X_test_adaboost, y_train_adaboost, y_test_adaboost = get_ngram_data(ngram_size=2, N=69446)
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


X_train_svm, X_test_svm, y_train_svm, y_test_svm = get_ngram_data(ngram_size=2, N=69446)
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


# # Decision Tree

# In[ ]:


X_train_tree, X_test_tree, y_train_tree, y_test_tree = get_ngram_data(ngram_size=2, N=69446)
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


final_data = []


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

with open('Bigrams_Final_data5.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['Recall','Precision','F1-score'])
    writer.writerow([L_Recall,L_Precision,L_F1_score])
    writer.writerow([R_Recall, R_Precision,R_F1_score])
    writer.writerow([N_Recall,N_Precision, N_F1_score])
    writer.writerow([KNN_Recall,KNN_Precision,KNN_F1_score])
    writer.writerow([Adaboost_Recall,Adaboost_Precision,Adaboost_F1_score])
    writer.writerow([svm_Recall,svm_Precision,svm_F1_score])
    writer.writerow([decision_Recall,decision_Precision,decision_F1_score])
    
    


# In[ ]:




