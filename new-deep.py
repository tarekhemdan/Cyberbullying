

# # import libraries 
import pandas as pd
from pandas import DataFrame
import itertools
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from sklearn import cross_validation#    from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
from tensorflow import keras

# # Read the dataset



data = pd.read_csv('twitter_racism_parsed_dataset.csv')

texts = data['Text'].astype(str)
labels = data['oh_label'].astype(int)


# # text preprocessing



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




data=preparedatasets(data)




#data=preparedatasets(data)
data.dropna(inplace=True)
data.head()
data['word_count'] = data['Text'].apply(lambda x: len(str(x).split()))
#Remove 0 and 1 word_count posts
new_data=data[(data.word_count >1)]
new_data.describe()    
data=new_data




data=data.drop_duplicates( keep="first")




# Preprocess the data
texts = data['Text'].astype(str)
labels = data['oh_label'].astype(int)




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




# ---------- ML Algorithms-----------
#cl=MultinomialNB()
#cl=LogisticRegression()
#cl=SVC()
#cl=DecisionTreeClassifier()
#cl=RandomForestClassifier()




# ---------- Deep Learning Algorithms ----------

# ------------  LSTM --------------
model = keras.Sequential()
model.add(keras.layers.Embedding(5000, 32, input_length=text_data.shape[1]))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(1, activation='sigmoid'))




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




#Number of words in the dataset
def get_count(data):
    d2=[]
    tt=""
    for index,r in data.iterrows():
        d2.append((r['Text'] ))
        tt+=r['Text']
    return len(tt)  




#cl.fit(X_train_vectorized, y_train)

history=model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))




y_pred = cl.predict(vectorizer.transform(X_test))




# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)




# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:', cm)






