#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Cyberbullying-Detection/Cyberbullying-Detection-on-Social-Media-using-Deep-Learning-and-Conventional-Machine-learning/blob/main/BERT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:

# In[ ]:

"""
get_ipython().system('pip install ktrain')


# In[ ]:


get_ipython().system('pip install bert-for-tf2 >> /dev/null')
get_ipython().system('pip install sentencepiece >> /dev/null')
get_ipython().system('pip install keras-self-attention')
get_ipython().system('pip install tf')
get_ipython().system('pip install grpcio')



get_ipython().system('pip install --upgrade grpcio >> /dev/null')
get_ipython().system('pip install tqdm  >> /dev/null')


# In[ ]:
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf

import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report


sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# In[ ]:


print(tf.__version__)


# In[ ]:


data = pd.read_csv("youtube.csv")


#columns_titles = ["tweet","label_bullying"]
columns_titles = ["text_message","label_bullying"]
data = data.reindex(columns=columns_titles)


# In[ ]:



#columns_titles = ["tweet","label_bullying"]
columns_titles = ["text_message","label_bullying"]
data=data.reindex(columns=columns_titles)


# In[ ]:





# In[ ]:


data.tail()


# In[ ]:


print("Size of dataset: ", data.shape)
#print("Size of test dataset: ", data_test.shape)


# In[ ]:


data.head()


# In[ ]:


data.count()


# In[ ]:


def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y


# In[ ]:


def repeat_positives(old_x, old_y, repeats=2):
    new_x = []
    new_y = []

    # rebuild the X dataset
    for i in range(len(old_x)):
        new_x.append(old_x[i])
        new_y.append(old_y[i])

        # if the example is a positive examples, repeat it in the dataset
        if old_y[i] == 1:
            for j in range(repeats-1):
                new_x.append(old_x[i])
                new_y.append(old_y[i])

    return new_x, new_y


# In[ ]:


X = data[:]['text_message']
X


# In[ ]:


Y = data[:]['label_bullying']
Y


# In[ ]:


import random
shuffle_data(X,Y)
X,Y = repeat_positives(X,Y)


# In[ ]:


len(X)


# In[ ]:


x_train = X[:1034]
x_test = X[1034:]
y_train = Y[:1034]
y_test = Y[1034:]


# In[ ]:


# the train and test percentages of the usual MLs have to be the same as here

# x_train = data[:700]["text_message"]
# x_test = data[700:]["text_message"]
# y_train = data[:700]["label_bullying"]
# y_test = data[700:]["label_bullying"]

#x_train = data[:12000]["tweet"]
#x_test = data[12000:]["tweet"]
#y_train = data[:12000]["label_bullying"]
#y_test = data[12000:]["label_bullying"]


# In[ ]:





# In[ ]:


train = pd.DataFrame(
    {'text': x_train,
     'label': y_train
    })

test = pd.DataFrame(
    {'text': x_test,
     'label': y_test
    })


# In[ ]:


x_train[0:5]


# In[ ]:


x_train[-6:-1]


# In[ ]:


x_test[0:5]


# In[ ]:


x_test[-6:-1]


# In[ ]:


chart = sns.countplot(train.label, palette=HAPPY_COLORS_PALETTE)
plt.title("No. of tweets per each class in train data")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');


# In[ ]:


chart = sns.countplot(test.label, palette=HAPPY_COLORS_PALETTE)
plt.title("No. of tweets per each class in test data")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');


# **THE BERT START PROPERLY HERE**

# In[ ]:


#get_ipython().system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')


# In[ ]:


#get_ipython().system('unzip uncased_L-12_H-768_A-12.zip')


# In[ ]:


os.makedirs("model2", exist_ok=True)


# In[ ]:


#get_ipython().system('mv uncased_L-12_H-768_A-12/ model2')


# In[ ]:


bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("model2/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


# #**Preprocessing**

# In[ ]:


class IntentDetectionData:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "label"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)


# In[ ]:


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))


# In[ ]:


tokenizer.tokenize("I can't wait to visit Kingsley again!")


# In[ ]:


tokens = tokenizer.tokenize("I can't wait to visit Kingsley again!")
tokenizer.convert_tokens_to_ids(tokens)


# In[ ]:


def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=512, activation="relu")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=256, activation="relu")(logits)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=256, activation="relu")(logits)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=128, activation="relu")(logits)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)


  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model


# #**Training**

# In[ ]:


classes = train.label.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)


# In[ ]:


data.max_seq_len


# In[ ]:


model = create_model(data.max_seq_len, bert_ckpt_file)


# In[ ]:


model.summary()


# In[ ]:


model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)


# In[ ]:


log_dir = "log/cyberbulling/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
  x=data.train_x, 
  y=data.train_y,
  validation_data= ([data.test_x], [data.test_y]),
  batch_size=128,
  shuffle=True,
  epochs=50,
  callbacks=[tensorboard_callback]
)


# #**Evaluation**

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


#get_ipython().run_line_magic('tensorboard', '--logdir log')


# In[ ]:


ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();


# In[ ]:


ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();


# In[ ]:


_, train_acc = model.evaluate(data.train_x, data.train_y)
_, test_acc = model.evaluate(data.test_x, data.test_y)

print("train acc", train_acc)
print("test acc", test_acc)


# In[ ]:


y_pred = model.predict(data.test_x).argmax(axis=-1)


# In[ ]:


print(classification_report(data.test_y, y_pred, target_names=classes))


# In[ ]:


import sklearn.metrics as sm
cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
print(cm) 
print(df_cm)

print("recall:", round(sm.recall_score(data.test_y, y_pred), 4))
print("precision:", round(sm.precision_score(data.test_y, y_pred), 4))
print("f1 score:", round(sm.f1_score(data.test_y, y_pred), 4))

# Normalize CM
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(norm_cm)

df_norm_cm = pd.DataFrame(norm_cm, index=classes, columns=classes)
#print(df_norm_cm)


# In[ ]:


hmap = sns.heatmap(df_norm_cm, annot=True, fmt="0.1f")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');


# In[ ]:


sentences = [
  "What's your age?",
  "fuck"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()


# In[ ]:





# In[ ]:





# In[ ]:




