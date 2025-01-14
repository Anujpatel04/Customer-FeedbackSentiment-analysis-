import streamlit as st
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from flask import Flask
import mysql.connector


dataset=pd.read_csv(r"C:\Users\a\Desktop\AI-NARESHIT\JANUARY\2nd - NLP project\2nd - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv",delimiter = '\t', quoting = 3)
    
# print(df)\

corpus=[]
for i in range(0, 2001):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv=TfidfVectorizer()    
# print(corpus)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
lb=LabelEncoder()
y=lb.fit_transform(y)
# print(x)
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.2)

classifier=XGBClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

def in_string(string):
    review = re.sub('[^a-zA-Z]', ' ',string)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

string=input('Enter a string here::')
processed_string=in_string(string)

result = cv.transform([processed_string]).toarray()
ans = classifier.predict(result)
print('predicted sentiment:','Possitive' if ans[0]==1 else 'Negative')
try:
    conn=mysql.connector.connect(
    host="localhost",
    user='root',
    password='root',
    database='j_db'
    )
    cursor=conn.cursor()
    if conn.is_connected():
        print('Connected to sql')
        cursor.execute('select * from country')
        for row in cursor.fetchall():
            print(row)

except Exception as e :
    print(f'error occured {e}')
