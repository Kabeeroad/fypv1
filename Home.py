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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from streamlit_gsheets import GSheetsConnection
from datetime import date

#loading ml model
@st.cache_resource
def load_model():
    with open('regression_model.joblib', 'rb') as joblib_in:
        model = joblib.load(joblib_in)
    return model

#function for getting data and making prediction
@st.cache_data() 
def predict(data):
    model = load_model()
    sentiment = get_sentiment(data)
    dlist = [data]
    topics = topic(dlist)
    data2 = pd.DataFrame([data], columns = ['Statement'])
    data2['Statement'] = preprocess(data2['Statement'])

    df1 = pd.read_csv('dataset/Lemm_df_part_1.csv',encoding='latin-1')
    df2 = pd.read_csv('dataset/Lemm_df_part_2.csv',encoding='latin-1')
    df3 = pd.read_csv('dataset/Lemm_df_part_3.csv',encoding='latin-1')
    df4 = pd.read_csv('dataset/Lemm_df_part_4.csv',encoding='latin-1')
    df5 = pd.read_csv('dataset/Lemm_df_part_5.csv',encoding='latin-1')
    df6 = pd.read_csv('dataset/Lemm_df_part_6.csv',encoding='latin-1')
    df7 = pd.read_csv('dataset/Lemm_df_part_7.csv',encoding='latin-1')
    df8 = pd.read_csv('dataset/Lemm_df_part_8.csv',encoding='latin-1')
    df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],ignore_index=True)
   
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
    classification=''
    if predictions == [1]:
        st.write(Real_Msg)
        classification+='Real'
    else:
        st.write (Fake_Msg)
        classification+='Fake'

    st.write(f'Additionally we found that this news article with the keywords of "{topics}" has a {sentiment} sentiment')
    datee=date.today()
    stuff=[data,classification,datee]
    old_data=conn.read()
    info=pd.DataFrame(data=[stuff],columns=['Article','Classification','Date'])
    concat_data = pd.concat([old_data, info], ignore_index=True)
    conn.update(data=concat_data)
    
#function for llm
def llm(text):
  pass 

#function for preprocessing data     
@st.cache_data(show_spinner=False)        
def preprocess(text):
  df = pd.DataFrame(text,columns=['Statement'])
  df['Statement'] = df['Statement'].str.replace(r'[^\x00-\x7f]_?', r'', regex=True)
  df['Statement'] = df['Statement'].str.replace(r'https?://\S+|www\.\S+', r'', regex=True)
  df['Statement'] = df['Statement'].str.replace(r'[^\w\s]', r'', regex=True)
  df['Statement'] = df['Statement'].apply(lambda x: word_tokenize(x))
  stop_words = set(stopwords.words('english'))
  df['Statement'] = df['Statement'].apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))
  lemmatizer = WordNetLemmatizer()
  df['Statement'] = df['Statement'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in x.split()]))
  text = df['Statement'].loc[0]
  return text

#function to get the sentiment of news
@st.cache_data(show_spinner=False)        
def get_sentiment(article):
  analyzer = SentimentIntensityAnalyzer()
  score = analyzer.polarity_scores(article)
  compound_score = score.get('compound')
  values = ['Positive', 'Neutral', 'Negative']
  rating = ''
  if compound_score >= 0.5:
    rating = values[0]
  elif (compound_score > - 0.5) and (compound_score < 0.5):
    rating = values[1]
  elif compound_score <= -0.5:
    rating = values[2]
  return rating

#topic modelling
@st.cache_data(show_spinner=False)        
def topic(article):
  text = [preprocess(article)]
  count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
  x_counts = count_vect.fit_transform(text)
  x_counts.todense()  
  tfidf_transformer = TfidfTransformer()
  x_tfidf = tfidf_transformer.fit_transform(x_counts)
  dimension = 1
  lda = LDA(n_components = dimension)
  lda_array = lda.fit_transform(x_tfidf)
  components = [lda.components_[i] for i in range(len(lda.components_))]
  features = list(count_vect.get_feature_names_out())
  important_words = [sorted(features, key = lambda x: components[j][features.index(x)], reverse = True)[:3] for j in range(len(components))]
  words=''
  c=0
  for i in important_words:
    for y in i:
      c+=1
      if c==1:
        words+=y+', '
      elif c==2:
        words+=y+' and '
      else:
        words+=y
  return words


#the actual website
st.set_page_config(
    page_title="🏠Home",
    page_icon="🕵️‍♂️",
)
conn = st.connection("gsheets",type=GSheetsConnection)


st.write("# Fake News Detector 🕵️‍♂️")

text = st.text_input("Enter an Article here:", key="Article")
st.write('Hint💡: Try to enter as much of the news article contents as possible and to not include information that is not related to the article.')

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

    st.write('Disclaimer⚠️ Machine Learning is not 100 percent accurate and can make mistakes❗')

    st.write('Five most recent Articles with their classification')
    sql='SELECT* FROM Sheet1 LIMIT 5' #uses sql select statement to get updated result
    select=conn.query(sql=sql,ttl=20)
    st.dataframe(select)
    



#Streamlit Website

    