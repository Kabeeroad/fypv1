import streamlit as st
import time
import pandas as pd
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


@st.cache_resource
def load_model():
    with open('regression_model.joblib', 'rb') as joblib_in:
        model = joblib.load(joblib_in)
    return model

@st.cache_data 
def predict(data):
    model = load_model()
    data2 = pd.DataFrame([data], columns = ['Statement'])
    data2['Statement'] = preprocess(data2['Statement'])

    df = pd.read_csv('Lemm_df.csv',encoding='latin-1')
    df = df.dropna()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Statement']) #instead of transforming each time could load transformed one 

    new_tfidf = vectorizer.transform(data2['Statement'])
    
    prediction = model.predict(new_tfidf)
    probabilities = model.predict_proba(new_tfidf)
    predictions = model.predict(new_tfidf)

    probabilities=list(probabilities)


    Fake = probabilities[0][0]
    Real = probabilities[0][1]
    Fake = round(Fake*100,1)
    Real = round(Real*100,1)



    Real_Msg = f'REAL! We predicted that the probability this News article is Real is {Real} percent'
    Fake_Msg = f'FAKE! We predicted that the probability this News article is Fake is {Fake} percent'

    if predictions == [1]:
        st.write(Real_Msg)
    else:
        st.write (Fake_Msg)

    
        
@st.cache_data        
def preprocess(text):
    text = text.str.replace(r'[^\x00-\x7f]_?', r'', regex=True)
    text = text.str.replace(r'https?://\S+|www\.\S+', r'', regex=True)
    text = text.str.replace(r'[^\w\s]', r'', regex=True)
    text = text.apply(lambda x: word_tokenize(x))
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in x.split()]))
    return text

    

st.set_page_config(
    page_title="Home",
    page_icon="🕵️‍♂️",
)

st.write("# Fake News Detector 🕵️‍♂️")

#st.sidebar.success("Select a demo above.")

text = st.text_input("Enter an Article", key="Article")

if text:
    st.write('Starting a long computation...')
    
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        latest_iteration.text(f'Analysing Text 🔎 {i+1}%')
        bar.progress(i + 1)
        
        time.sleep(0.05)  

    st.write("Analysis Complete")
    predict(text)

    