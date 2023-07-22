import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import time
import nltk


# # import libraries 
from pandas import DataFrame
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk

# Download the stopwords corpus
nltk.download('stopwords')
# Download the Punkt tokenizer data
nltk.download('punkt')

#from sklearn import cross_validation#    from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()



# # Read the dataset

# In[203]:


data = pd.read_csv('aggression_parsed_dataset.csv')

Text = data['Text'].astype(str)
oh_label = data['oh_label'].astype(int)


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
Text = data['Text'].astype(str)
oh_label = data['oh_label'].astype(int)



# Load pre-trained word embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Define the focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_loss = - alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return focal_loss_fn

# Load the dataset
#data = pd.read_csv('aggression_parsed_dataset.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['Text'].values)
train_sequences = tokenizer.texts_to_sequences(train_data['Text'].values)
test_sequences = tokenizer.texts_to_sequences(test_data['Text'].values)

# Pad the sequences to the same length
train_padded = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')

# Load pre-trained word embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_dim = 100
word_index = tokenizer.word_index
num_words = min(10000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=100, weights=[embedding_matrix], input_length=100, trainable=False),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with the focal loss function
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=focal_loss(), metrics=['accuracy'])

# Train the model
start_time = time.time()
history = model.fit(train_padded, train_data['oh_label'].values, epochs=20, batch_size=32, validation_split=0.2)
elapsed_time_train = time.time() - start_time
print('Time taken for training:', elapsed_time_train, 'seconds')

# Evaluate the model on the testing set
start_time = time.time()
y_pred = model.predict(test_padded)
elapsed_time_test = time.time() - start_time
print('Time taken for testing:', elapsed_time_test, 'seconds')

# Round the predictions to the nearest integer
y_pred = np.round(y_pred).flatten()
y_true = test_data['oh_label'].values

# Print various metrics
print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))
print('\nClassification report:')
print(classification_report(y_true, y_pred))
print('\nAccuracy:', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred))
print('Recall:', recall_score(y_true, y_pred))
print('F1 score:',f1_score(y_true, y_pred))

# Calculate and plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print('Area under ROC curve:', roc_auc)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()