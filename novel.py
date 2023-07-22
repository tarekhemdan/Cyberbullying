# Load the dataset
import pandas as pd

data = pd.read_csv('kaggle_parsed_dataset.csv')

# Check if the 'bullying_trace' column is present in the dataset
if 'bullying_trace' not in data.columns:
    raise ValueError('The dataset does not have a "bullying_trace" column')

# Preprocess the data

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data['Text'], data['bullying_trace'], test_size=0.2)

# Tokenize the text data
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data)

train_seq = tokenizer.texts_to_sequences(train_data)
test_seq = tokenizer.texts_to_sequences(test_data)

# Pad the sequences
from keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_seq, maxlen=100)
test_seq = pad_sequences(test_seq, maxlen=100)

# Build the model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(10000, 32, input_length=100))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_seq, train_labels, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(test_seq, test_labels)
predictions = model.predict(test_seq)

# Calculate other metrics such as recall, precision, and F1 score
from sklearn.metrics import classification_report

report = classification_report(test_labels, predictions.round())
print(report)
