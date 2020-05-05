#!/usr/bin/env python
# coding: utf-8

# # Sentiment Classification with Natural Language Processing on LSTM 

# In[ ]:


# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras.backend as K
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# dirpath = 'C:/Users/bhumi/Documents/SearchFinalProject/dataset'
dirpath = '/geode2/home/u010/bagrawal/Carbonate/SearchFinalProject/dataset'
review_iter = pd.read_json(dirpath + '/review.json',chunksize = 10000,lines=True)
review_df = pd.DataFrame()
i=0
for df in review_iter:
    if i < 10:
        review_df = pd.concat([review_df, df])
        i += 1
    else:
        break


# In[ ]:


print(review_df.shape)


# In[ ]:


print(review_df.head())


# 
# # Text Cleaning or Preprocessing

# In[ ]:


review_data = pd.DataFrame()
label_data = pd.DataFrame()
review_data["text"] = review_df["text"].str.lower().replace('[^\w\s]','')
label_data["label"] = review_df['stars']


# In[ ]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


review_list = review_data["text"].tolist()


# In[ ]:


corpus = []
for review in review_list:
    # review = review.split()
    # ps = PorterStemmer()
    # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # review = ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus=pd.DataFrame(corpus, columns=['Reviews']) 
corpus.head()


# In[ ]:


result=corpus.join(review_df['stars'])
result.head()


# # TFIDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(result['Reviews'])


# In[ ]:


X = tfidf.transform(result['Reviews'])
result['Reviews'][1]


# In[ ]:


print([X[1, tfidf.vocabulary_['good']]])


# In[ ]:


print([X[1, tfidf.vocabulary_['old']]])


# In[ ]:


print([X[1, tfidf.vocabulary_['down']]])


# # Sentiment Classification

# In[ ]:


type(result)


# In[ ]:





# In[ ]:


result.dropna(inplace=True)
result[result['stars'] != 3]
result['Positivity'] = np.where(result['stars'] > 3, 1, 0)
cols = [ 'stars']
result.drop(cols, axis=1, inplace=True)
result.head()


# In[ ]:


result.groupby('Positivity').size()


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X = result.Reviews
y = result.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[ ]:


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),
                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))


# In[ ]:


print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),
                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# In[ ]:


def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


# In[ ]:


cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,25001,5000)

def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Test result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()


# In[ ]:


print("Result for trigram with stop words (Tfidf)\n")
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))


# In[ ]:


from sklearn.metrics import classification_report

cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))
pipeline = Pipeline([
        ('vectorizer', cv),
        ('classifier', rf)
    ])
sentiment_fit = pipeline.fit(X_train, y_train)
y_pred = sentiment_fit.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['negative','positive']))


# In[ ]:


## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = pipeline, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))


# # Chi2 Feature Selection

# In[ ]:


from sklearn.feature_selection import chi2

tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))
X_tfidf = tfidf.fit_transform(result.Reviews)
y = result.Positivity
chi2score = chi2(X_tfidf, y)[0]


# In[ ]:


plt.figure(figsize=(16,8))
scores = list(zip(tfidf.get_feature_names(), chi2score))
chi2 = sorted(scores, key=lambda x:x[1])
topchi2 = list(zip(*chi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show();


# # LSTM neural network

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[ ]:


max_fatures = 30000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(result['Reviews'].values)
X1 = tokenizer.texts_to_sequences(result['Reviews'].values)
X1 = pad_sequences(X1)


# In[ ]:


Y1 = pd.get_dummies(result['Positivity']).values
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)
print(X1_train.shape,Y1_train.shape)
print(X1_test.shape,Y1_test.shape)


# In[ ]:
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

embed_dim = 150
lstm_out = 200

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = [recall_m,precision_m,f1_m])
print(model.summary())


# In[ ]:


batch_size = 32
model.fit(X1_train, Y1_train, nb_epoch = 5, batch_size=batch_size, verbose = 2)


# In[ ]:


recall,precision,f1_score = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)
print("recall: %.2f" % (recall))
print("precision: %.2f" % (precision))
print("f1 score: %.2f" % (f1_score))
