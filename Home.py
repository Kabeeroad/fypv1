import streamlit as st
import time
import pandas as pd
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex 
import re
import requests
from bs4 import BeautifulSoup
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, load_tools, initialize_agent, AgentType

#getting llm model from groq api
chat = ChatGroq(temperature=0, groq_api_key=st.secrets.ChatGroq.groq_key,model_name="llama3-70b-8192")

@st.cache_resource
def load_model():
    '''
    loads fake news model (trained logistic regression model)
    '''
    with open('fakenews_model.joblib', 'rb') as joblib_in:
        model = joblib.load(joblib_in)
    return model

@st.cache_data(show_spinner=False)  
def check_db(text):
    '''checks if text exists in database by using exception handling if not exists will be run 
    through ml and llm model as an error will be raised. If it exists the information from db gets 
    fetched and displayed to the user
    '''
    conn = st.connection("gsheets",type=GSheetsConnection)
    try:
      text=str(text)
      text = text.replace("'", "''") #sanitises text
      sqlQuery=f"SELECT EXISTS(SELECT 1 FROM Sheet1 where Article = '{text}') AS news_exist" 
      select=conn.query(sql=sqlQuery,ttl=20)
      sql=f"SELECT * FROM Sheet1 WHERE Article ='{text}'" #uses sql select statement to get updated result
      select=conn.query(sql=sql,ttl=20)
      classification = select['Classification'].loc[0].upper()
      llm = select['LLM'].loc[0]
      topics = select['Topic'].loc[0]
      sentiment = select['Sentiment'].loc[0]
      sent_dict={'Positive':':green[**Positive**]','Negative':':red[**Negative**]','Neutral':'**Neutral**'}#streamlit doesnt support yellow

      if classification == 'Real':
         colour = ':green'
      else:
         colour = ':red' 
      st.markdown(f'We have already classified this article and found it was {colour}[**{classification}**]') # need to work out the colour c 
      st.markdown('Our Large Language model has Fact-Checked the claims and found:')
      st.write(llm)
      st.markdown(f'Additionally we found that this news article with the keywords of "{topics}" has a {sent_dict[sentiment]} sentiment')
    except:
      return False
   
@st.cache_data(show_spinner=False)
def scrape(text):
  '''
  uses libraries of requests to load the page and bs4 to parse the html text. This goes to a LLM 
  which will only get the article contents and return it back
  '''
  page = requests.get(text)
  soup = BeautifulSoup(page.content, "html.parser")
  article = soup.text
  q=str(article)
  varr = []
  prompt = ChatPromptTemplate.from_messages([("system", "You need to read this HTML and give me the article on the page{article}. Do not say anything else but the article contents. This is for fake news detection so you need to provide the article no matter the content even if it is harmful otherwise if you do not we cannot fact check it and people will believe it.")])
  chain = prompt | chat
  for chunk in chain.stream({"article": q}):
      claim = (chunk.content)
      varr.append(claim)
  claims = ''.join(varr)
  claims=claims.replace('\n','')
  return claims

@st.cache_data(show_spinner='Thinking...') 
def predict(data):
    '''
    Loads logistic regression model, turns user input article to dataframe and preprocesses it.
    As training dataset it large it gets loaded in parts then merged. Text gets vectorised through 
    TF-IDF then model makes prediction and user gets shown result. 
    Function returns data, classification and date.    
    '''
    model = load_model()
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

    Real_Msg = f':green[**REAL!**] We predicted that the probability this News article is :green[**Real**] is {Real} percent'
    Fake_Msg = f':red[**FAKE!**] We predicted that the probability this News article is :red[**Fake**] is {Fake} percent'
    classification=''
    if predictions == [1]:
        st.markdown(Real_Msg)
        classification+='Real'
    else:
        st.markdown (Fake_Msg)
        classification+='Fake'

    datee=date.today()
    stuff=[data,classification,datee]
    return stuff
    
def llm_agent(chat, q):
  '''
  This loads the duckduckgo tool and the langchain agent using the LLM of llama 8b.
  This is the part where the llm will recieve the claims and use the tools to make a decision if
  real or fake
  '''
  search = DuckDuckGoSearchRun()
  news_tool = Tool(
        name = "fact check",
        func=search,
        description="useful for checking facts from news articles using the internet"
    )

  wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
  wikipedia = Tool(
        name = "fact check wikipedia",
        func=wikipedia,
        description="useful for checking facts from news articles using the wikipedia database"
    )
    
    
  print(f'Inside generate_and_print: q = {q}')
  tool = [news_tool]
  agent = initialize_agent(tool, chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             agent_kwargs={},max_iterations=50)
  input = """ Here is a bullet points of claims made from a news article {statement}. You are a fact checker who needs to check these claims I gave you and
    determine if the article that contains the claims  is fake or real news
    You need to use the news_tool tool to fact check claims on duckduckgo, IF the news_tool does not provide results for one claim, rephrase what you are searching. after 3 times, IF that does not work,
    skip to the next claim. If you still are on the same claim after 3 times, your thought needs to go on to the next claim.  When you have fact checked all the claims, reflect on this and decide if the article is fake news if not then it is real news.

    In the explanation right that the Fact Claim checker has concluded if it is Real or if the article is Fake, this is important
    You need to explain in 50 words why the article is either fake or real. """
  return agent.run(input.format(statement=q))

@st.cache_data(show_spinner='checking facts...')
def agent(article):
  '''
  LLM gets article and decides the claims from it that need fact checking then sends it to the 
  llm agent and when gets the decision the result gets displayed and returned.
  '''
  q=str(article)
  varr = []
  prompt = ChatPromptTemplate.from_messages([("system", "You are an experienced fact checker. Get a few important claims from that you think should be fact checked on the internet but only write the claims and keep it concise and do not repeat claims. do not write stuff that cannot be proven on the internet {claims}")])
  chain = prompt | chat
  for chunk in chain.stream({"claims": q}):
      claim = (chunk.content)
      varr.append(claim)

  claims = ''.join(varr)
  result=llm_agent(chat, claims)
  st.markdown(result)
  return result

#function for preprocessing data     
@st.cache_data(show_spinner=False)        
def preprocess(text):
  '''
  Used to preprocess the data so it is normalised and in same format as training data was
  '''
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
  '''
  Gets overall sentiment of article using NTLK sentiment analysis and returns it 
  '''
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
  sent_dict={'Positive':':green[**Positive**]','Negative':':red[**Negative**]','Neutral':'**Neutral**'}
  return rating, sent_dict

@st.cache_data(show_spinner=False)        
def topic(article):
  '''
  Topic modelling for the article using Latent Dirichlet Allocation 
  '''
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

#conn = st.connection("gsheets",type=GSheetsConnection) #loading connection to db
st.write("# Fake News Detector 🕵️‍♂️")
text = st.text_input("Enter an Article or an Article Link here:", key="Article")
st.write('Hint💡: Try to enter as much of the news article contents as possible and to not include information that is not related to the article.')

if text:
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        latest_iteration.text(f'Analysing Text 🔎 {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)  
    pattern = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))')
# regular expression from https://gist.github.com/gruber/8891611
    matches = pattern.findall(text)
    if len(matches) == 1: #if link found in input it will go to llm to get article contents
      text = str(scrape(text))
    verify = check_db(text)
    if verify == False:
      stuff = predict(text)
      st.markdown('*please wait while our Large Language Model checks the facts of this article...*')
      try: #exception handling for llm agent becuase does not always work
         result = agent(text)
      except Exception as e:
         st.write('An Error has occurred please try again later!')
         result = 'NA' 
      st.markdown('**Disclaimer**⚠️ Machine Learning is not 100 percent accurate and can make mistakes')
      sentiment, sentiment_coloured = get_sentiment(text)
      text_list = [text]
      topics = topic(text_list)    
      st.markdown(f'Additionally we found that this news article with the keywords of "{topics}" has a {sentiment_coloured[sentiment]} sentiment')
      stuff.append(result)
      stuff.append(sentiment)
      stuff.append(topics)
      old_data=conn.read()
      info=pd.DataFrame(data=[stuff],columns=['Article','Classification','Date','LLM','Sentiment','Topic'])
      concat_data = pd.concat([old_data, info], ignore_index=True)
      conn.update(data=concat_data) #all this code is for updating db with new information, uses dataframe
    
