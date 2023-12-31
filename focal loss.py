import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

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
data = pd.read_csv('aggression_parsed_dataset.csv')

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

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with the focal loss function
model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Train the model
history = model.fit(train_padded, train_data['oh_label'].values, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing set
y_pred = model.predict(test_padded)
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
print('F1 score:', f1_score(y_true, y_pred))