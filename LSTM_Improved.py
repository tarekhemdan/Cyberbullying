import pandas as pd
import itertools
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv('twitter_racism_parsed_dataset.csv')

# Text preprocessing
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
    for index, r in data.iterrows():
        Text = stopwordremoval(r['Text'])
        Text = stemming(r['Text'])
        sentences.append([Text, r['oh_label']])
    df_sentence = pd.DataFrame(sentences, columns=["Text", "oh_label"])
    return df_sentence

data = preparedatasets(data)
data.dropna(inplace=True)
data['word_count'] = data['Text'].apply(lambda x: len(str(x).split()))
new_data = data[(data.word_count > 1)]
new_data.describe()    
data = new_data
data = data.drop_duplicates(keep="first")

# Preprocess the data
texts = data['Text'].astype(str)
labels = data['oh_label'].astype(int)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
text_data = keras.preprocessing.sequence.pad_sequences(text_sequences)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2)

# Define and compile the LSTM model
model = keras.Sequential()
model.add(keras.layers.Embedding(5000, 64, input_length=text_data.shape[1]))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_proba = model.predict(X_test)
y_pred = np.round(y_pred_proba)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion)
print("ROC AUC Score:", roc_auc)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
