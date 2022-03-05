# -*- coding: utf-8 -*-
"""Spam Classifier.ipynb

Original file is located at
    https://colab.research.google.com/drive/1D46_ZxjyLJ8wLlleJ4y-lXML60jy6YoE
"""

import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')

data = pd.read_csv('https://techlearn-cdn.s3.amazonaws.com/project_GmailSpamClassification/spam.csv' , encoding='cp1252')

data.head()

data = data[['v1','v2']]

data['v1'] = data['v1'].apply(lambda x:0 if x=='ham' else 1 )

data

'''

'''

def process(x):
    temp = []
    document = nlp(x.lower())
    print(document)
    for i in document:
        if i.is_stop!=True and i.is_punct!= True:
            print(i)
            temp.append(i.lemma_)
            print(temp)
        else:
            pass
        
    return (' '.join(temp))

data['v2'] = data['v2'].apply(lambda x: process(x))



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')

text_vector = vectorizer.fit_transform(data['v2'].values.tolist())

print(text_vector)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(text_vector.toarray(), data['v1'], test_size=0.15, random_state=20)

len(x_test)

from sklearn.naive_bayes import BernoulliNB

modelB = BernoulliNB()
modelB.fit(x_train,y_train)
print(modelB.score(x_train,y_train))

y_predict = modelB.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))

