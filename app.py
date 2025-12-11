import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import porter
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_message(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)
    a = []
    for i in Message:
        if i.isalnum():
            a.append(i)
    Message = a[:]
    a.clear()
    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)
    Message = a[:]
    a.clear()
    for i in Message:
        a.append(ps.stem(i))
    return " ".join(a)
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
st.title('Email/SMS Spam Detector')
input_sms = (st.text_area('Enter the Message'))
if st.button('Predict'):
    transform_sms = transform_message(input_sms) #1 Preprocess
    vector_input  = tfidf.transform([transform_sms]) #2 Vectorize
    Prediction = model.predict(vector_input)[0] #3 Predict
    if Prediction == 1:#4 Display
        st.header("Spam")
    else:
        st.header("Not Spam")

