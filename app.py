
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#", '', text)
    text = re.sub(r"[^A-Za-z\s]", '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Dummy model & vectorizer - replace with your trained model
vectorizer = TfidfVectorizer()
model = MultinomialNB()
X = vectorizer.fit_transform(["good", "bad", "amazing", "terrible", "great", "worst"])
y = ["Positive", "Negative", "Positive", "Negative", "Positive", "Negative"]
model.fit(X, y)

st.title("Twitter Sentiment Analysis")
user_input = st.text_area("Enter a tweet")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    st.success(f"The sentiment is: {prediction}")
