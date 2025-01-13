# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure you download the required NLTK data
# nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\a\Desktop\AI-NARESHIT\JANUARY\2nd - NLP project\2nd - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
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
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
lb = LabelEncoder()
y = lb.fit_transform(y)

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the classifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Function to preprocess a single review
def preprocess_review(input_string):
    review = re.sub('[^a-zA-Z]', ' ', input_string)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    processed_review = ' '.join(review)
    return processed_review

# Streamlit app
st.set_page_config(page_title="Restaurant Review Sentiment Analyzer", layout="wide")

# Add custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://www.example.com/your-background-image.jpg'); /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        color: white;
    }
    .stApp {
        background-color: rgba(206, 196, 196, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit title and description
st.title("Restaurant Review Sentiment Analyzer")
st.write("Enter your review below, and we'll predict whether the sentiment is Positive or Negative!")

# Input for user's review
string = st.text_input("Enter your review:")

if string:
    # Preprocess the review
    processed_string = preprocess_review(string)

    # Transform the preprocessed string using the existing vectorizer
    result = cv.transform([processed_string]).toarray()

    # Predicting the sentiment
    ans = classifier.predict(result)
    
    # Display the sentiment
    st.subheader("Predicted Sentiment:")
    st.write("🌟 **Positive**" if ans[0] == 1 else "💔 **Negative**")
