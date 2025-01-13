# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
# Importing the dataset
dataset = pd.read_csv(r"C:\Users\a\Desktop\AI-NARESHIT\JANUARY\2nd - NLP project\2nd - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] 

for i in range(0, 2001):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
lb=LabelEncoder()
y=lb.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from xgboost import XGBClassifier
classifier=XGBClassifier()


#traning data
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



def preprocess_review(input_string):
    
    # Remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', input_string)
    
    # Convert text to lowercase
    review = review.lower()
    
    # Split text into words
    review = review.split()
    
    # Initialize PorterStemmer
    ps = PorterStemmer()
    
    # Remove stopwords and apply stemmingit was really nice visitng here
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    
    # Join the words back into a single string
    processed_review = ' '.join(review)
    
    return processed_review

#Streamlit title here:
st.title("Please Enter you reviw on our resturent here:")
# Predicting a Single Review
string=st.text_input("Enter your review: ")
processed_string = preprocess_review(string)
print("\nProcessed Review:", processed_string)


# Transform the preprocessed string using the existing vectorizer
result = cv.transform([processed_string]).toarray()  # Use transform, not fit_transform


# Predicting the sentiment
ans = classifier.predict(result)
st.write("\nPredicted Sentiment:", "Positive" if ans[0] == 1 else "Negative")

