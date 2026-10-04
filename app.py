import streamlit as st
import joblib

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab',quiet=True)
nltk.download("stopwords",quiet=True)
# Instantiate the stemmer outside or inside the function
ps = PorterStemmer()

model = joblib.load("model.pkl")
tfidf = joblib.load(open("vectorizer.pkl", "rb"))
st.title("EZEEDetect Welcome to SMS SPAM CLASSIFIER")

message = st.text_input("ENTER YOUR MESSAGE:")

# preprocessing
def transform_text(text):
    text = text.lower()

    # tokenize
    text = nltk.word_tokenize(text)

    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))  # <--- Call .stem() on the instance ps, not the class

    return y

if message:
    if not hasattr(tfidf, "vocabulary_") or not hasattr(tfidf, "idf_"):
        st.error(
            "The saved vectorizer is not fitted. Replace vectorizer.pkl with the fitted "
            "TF-IDF vectorizer used to train model.pkl."
        )
        st.stop()

    transformed_text = transform_text(message)
    vector_input = tfidf.transform([" ".join(transformed_text)])
    result = model.predict(vector_input)[0]
    st.header("SPAM" if result == 1 else "NOT SPAM")