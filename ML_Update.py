# Import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# Read the dataset
data = pd.read_csv('twitter_racism_parsed_dataset.csv')

# Preprocessing functions
def stopword_removal(text):
    stop = stopwords.words('english')
    needed_words = []
    words = word_tokenize(text)
    for w in words:
        if len(w) >= 2 and w not in stop:
            needed_words.append(w)
    filtered_sent = " ".join(needed_words)
    return filtered_sent

def stemming(text):
    st = ISRIStemmer()
    stemmed_words = []
    words = word_tokenize(text)
    for w in words:
        stemmed_words.append(st.stem(w))
    stemmed_sent = " ".join(stemmed_words)
    return stemmed_sent

def prepare_datasets(data):
    sentences = []
    for index, r in data.iterrows():
        text = stopword_removal(r['Text'])
        text = stemming(text)
        sentences.append([text, r['oh_label']])
    df_sentence = pd.DataFrame(sentences, columns=["Text", "oh_label"])
    return df_sentence

# Prepare the datasets
data = prepare_datasets(data)
data.dropna(inplace=True)
data = data[(data['Text'].str.len() > 1)]
data.drop_duplicates(keep="first", inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['oh_label'], test_size=0.2, random_state=0)

# Vectorize the training data
vectorizer = TfidfVectorizer(analyzer='word', smooth_idf=True, ngram_range=(1,2))
vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train_vectorized, y_train)

# Predict on the test set and calculate evaluation metrics
y_pred = clf.predict(vectorizer.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Calculate ROC and AUC
y_prob = clf.predict_proba(vectorizer.transform(X_test))[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Print the evaluation metrics and ROC curve
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Confusion Matrix:\n', cm)
print('AUC: {:.2f}'.format(auc))

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
