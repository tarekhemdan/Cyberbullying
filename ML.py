
# # import libraries 
import pandas as pd
from pandas import DataFrame
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import cross_validation#    from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *


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





X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['oh_label'],test_size=0.20, random_state = 0)
vectorizer = TfidfVectorizer( analyzer='word',smooth_idf=True, ngram_range=(1,2))
vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)





#cl=MultinomialNB()
#cl=LogisticRegression()
#cl=SVC()
#cl=DecisionTreeClassifier()
cl=RandomForestClassifier()




cl.fit(X_train_vectorized, y_train)





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







