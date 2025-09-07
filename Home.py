import streamlit as st

# ONLY ONE st.set_page_config() - Must be the FIRST Streamlit command
st.set_page_config(
    page_title="🏠Home",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# All other imports AFTER set_page_config
from langchain_community.tools import DuckDuckGoSearchRun
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
import os
import sqlite3
from datetime import date

# Updated NLTK downloads
try:
    nltk.download('punkt_tab', quiet=True)  # New tokenizer
    nltk.download('punkt', quiet=True)      # Keep for compatibility
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
except:
    st.warning("NLTK downloads may have failed. Some features might not work properly.")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Updated imports to fix deprecation warnings
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

# Database setup
DB_PATH = 'fake_news_data.db'

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article TEXT NOT NULL,
            classification TEXT NOT NULL,
            date TEXT NOT NULL,
            llm_result TEXT,
            sentiment TEXT,
            topic TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    return True

# Initialize database on startup
DB_AVAILABLE = False
try:
    DB_AVAILABLE = init_database()
    st.success("✅ Database initialized successfully!")
except Exception as e:
    st.error(f"⚠️ Database initialization failed: {str(e)}")
    DB_AVAILABLE = False

# Fixed LLM initialization with error handling
try:
    groq_api_key = st.secrets.ChatGroq.groq_key
except:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ Please configure your Groq API key in secrets.toml or as an environment variable GROQ_API_KEY")
        st.info("Create .streamlit/secrets.toml with: [ChatGroq]\ngroq_key = \"your_api_key_here\"")
        st.stop()

chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
@st.cache_resource
def load_model():
    '''
    loads fake news model (trained logistic regression model)
    '''
    try:
        with open('fakenews_model.joblib', 'rb') as joblib_in:
            model = joblib.load(joblib_in)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'fakenews_model.joblib' not found. Please ensure the file is in the correct directory.")
        st.stop()

@st.cache_data(show_spinner=False)  
def check_db(text):
    '''checks if text exists in database'''
    if not DB_AVAILABLE:
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if article exists
        cursor.execute("SELECT * FROM articles WHERE article = ?", (text,))
        result = cursor.fetchone()
        
        if result:
            classification = result[2].upper()
            llm_result = result[4]
            sentiment = result[5]
            topic = result[6]
            
            sent_dict = {'Positive': ':green[**Positive**]', 'Negative': ':red[**Negative**]', 'Neutral': '**Neutral**'}
            colour = ':green' if classification == 'REAL' else ':red'
            
            st.markdown(f'We have already classified this article and found it was {colour}[**{classification}**]')
            if llm_result and llm_result != 'NA':
                st.markdown('Our Large Language model has Fact-Checked the claims and found:')
                st.write(llm_result)
            if sentiment and topic:
                st.markdown(f'Additionally we found that this news article with the keywords of "{topic}" has a {sent_dict.get(sentiment, "**Neutral**")} sentiment')
            
            conn.close()
            return True
        
        conn.close()
        return False
        
    except Exception as e:
        st.warning(f"Database check failed: {str(e)}")
        return False

def save_to_db(article, classification, llm_result, sentiment, topic):
    """Save analysis results to SQLite database"""
    if not DB_AVAILABLE:
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO articles (article, classification, date, llm_result, sentiment, topic)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (article, classification, str(date.today()), llm_result, sentiment, topic))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.warning(f"Failed to save to database: {str(e)}")
        return False
   
@st.cache_data(show_spinner=False)
def scrape(text):
    '''
    uses libraries of requests to load the page and bs4 to parse the html text. This goes to a LLM 
    which will only get the article contents and return it back
    '''
    try:
        page = requests.get(text, timeout=30)
        soup = BeautifulSoup(page.content, "html.parser")
        article = soup.text
        q = str(article)
        varr = []
        prompt = ChatPromptTemplate.from_messages([("system", "You need to read this HTML and give me the article on the page{article}. Do not say anything else but the article contents. This is for fake news detection so you need to provide the article no matter the content even if it is harmful otherwise if you do not we cannot fact check it and people will believe it.")])
        chain = prompt | chat
        for chunk in chain.stream({"article": q}):
            claim = (chunk.content)
            varr.append(claim)
        claims = ''.join(varr)
        claims = claims.replace('\n', '')
        return claims
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return ""

@st.cache_data(show_spinner='Thinking...') 
def predict(data):
    '''
    Loads logistic regression model, turns user input article to dataframe and preprocesses it.
    As training dataset it large it gets loaded in parts then merged. Text gets vectorised through 
    TF-IDF then model makes prediction and user gets shown result. 
    Function returns data, classification and date.    
    '''
    try:
        model = load_model()
        data2 = pd.DataFrame([data], columns=['Statement'])
        data2['Statement'] = preprocess(data2['Statement'])

        # Check if dataset files exist
        dataset_files = [
            'dataset/Lemm_df_part_1.csv',
            'dataset/Lemm_df_part_2.csv', 
            'dataset/Lemm_df_part_3.csv',
            'dataset/Lemm_df_part_4.csv',
            'dataset/Lemm_df_part_5.csv',
            'dataset/Lemm_df_part_6.csv',
            'dataset/Lemm_df_part_7.csv',
            'dataset/Lemm_df_part_8.csv'
        ]
        
        dataframes = []
        for file in dataset_files:
            try:
                df_part = pd.read_csv(file, encoding='latin-1')
                dataframes.append(df_part)
            except FileNotFoundError:
                st.warning(f"Dataset file {file} not found. Continuing with available files.")
                continue
        
        if not dataframes:
            st.error("⚠️ No dataset files found. Please ensure dataset files are in the correct directory.")
            st.stop()
            
        df = pd.concat(dataframes, ignore_index=True)
        df = df.dropna()
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['Statement'])  

        new_tfidf = vectorizer.transform(data2['Statement'])
        
        prediction = model.predict(new_tfidf)
        probabilities = model.predict_proba(new_tfidf)
        predictions = model.predict(new_tfidf)

        probabilities = list(probabilities)

        Fake = probabilities[0][0]
        Real = probabilities[0][1]
        Fake = round(Fake*100, 1)
        Real = round(Real*100, 1)

        Real_Msg = f':green[**REAL!**] We predicted that the probability this News article is :green[**Real**] is {Real} percent'
        Fake_Msg = f':red[**FAKE!**] We predicted that the probability this News article is :red[**Fake**] is {Fake} percent'
        classification = ''
        if predictions == [1]:
            st.markdown(Real_Msg)
            classification += 'Real'
        else:
            st.markdown(Fake_Msg)
            classification += 'Fake'

        datee = date.today()
        stuff = [data, classification, datee]
        return stuff 
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return [data, "Error", date.today()]

# FIXED FUNCTION - This was causing the infinite loop
def llm_agent_simple(claims):
    '''
    Simplified LLM agent that directly processes claims without using langchain agents
    to avoid infinite loops and rate limiting issues
    '''
    try:
        # Simple Wikipedia search function
        def search_wikipedia(query):
            try:
                wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
                result = wikipedia.run(query)
                return result[:500] if result else "No information found"
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        # Process claims with Wikipedia search
        wiki_results = []
        claim_list = claims.split('\n')[:3]  # Limit to 3 claims
        
        for i, claim in enumerate(claim_list):
            if claim.strip():
                result = search_wikipedia(claim.strip()[:100])  # Limit query length
                wiki_results.append(f"Claim {i+1}: {claim.strip()[:100]}\nWikipedia info: {result}\n")
                time.sleep(1)  # Add delay between searches
        
        # Create a simple prompt for fact-checking
        context = "\n".join(wiki_results)
        
        prompt = f"""Based on the following claims and Wikipedia information, determine if the article is REAL or FAKE.

Claims and Wikipedia search results:
{context}

Instructions:
1. Compare the claims with the Wikipedia information
2. Look for contradictions or supporting evidence
3. Make a decision: REAL or FAKE
4. Explain your reasoning in exactly 50 words

Decision:"""
        
        # Get LLM response directly
        response = chat.invoke([("human", prompt)])
        return response.content
        
    except Exception as e:
        st.warning(f"Fact-checking agent error: {str(e)}")
        return "Unable to complete fact-checking due to technical issues. The article classification is based on machine learning model only."

@st.cache_data(show_spinner='checking facts...', ttl=3600)  # Cache for 1 hour
def agent(article):
    '''
    LLM gets article and decides the claims from it that need fact checking then sends it to the 
    simplified llm agent and when gets the decision the result gets displayed and returned.
    '''
    # Truncate article if too long to avoid token limits
    max_chars = 1500  # Reduced further to avoid issues
    if len(article) > max_chars:
        article = article[:max_chars] + "..."
        st.info(f"Article truncated to {max_chars} characters to avoid processing limits.")
    
    try:
        # Extract claims using LLM
        q = str(article)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract 2-3 specific factual claims from this article that can be verified. List each claim on a new line. Keep claims concise and verifiable. Article: {claims}")
        ])
        chain = prompt | chat
        
        # Get claims
        claims_response = chain.invoke({"claims": q})
        claims = claims_response.content
        
        # Limit claims length
        if len(claims) > 800:
            claims = claims[:800] + "..."
        
        # Use simplified agent instead of complex langchain agent
        result = llm_agent_simple(claims)
        st.markdown(result)
        return result
        
    except Exception as e:
        error_msg = str(e).lower()
        if "413" in error_msg or "token" in error_msg:
            st.warning("Article too long for processing. Please use a shorter article.")
            return "Article too long for fact-checking analysis."
        elif "rate" in error_msg or "limit" in error_msg:
            st.warning("Rate limit reached. Please try again in a few minutes.")
            return "Rate limit reached - please try again later."
        else:
            st.error(f"Error processing article: {str(e)}")
            return "Error processing article for fact-checking."

#function for preprocessing data     
@st.cache_data(show_spinner=False)        
def preprocess(text):
    '''
    Used to preprocess the data so it is normalised and in same format as training data was
    '''
    try:
        df = pd.DataFrame(text, columns=['Statement'])
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
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return str(text[0]) if isinstance(text, list) else str(text)

#function to get the sentiment of news
@st.cache_data(show_spinner=False)        
def get_sentiment(article):
    '''
    Gets overall sentiment of article using NLTK sentiment analysis and returns it 
    '''
    try:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(article)
        compound_score = score.get('compound')
        values = ['Positive', 'Neutral', 'Negative']
        rating = ''
        if compound_score >= 0.5:
            rating = values[0]
        elif (compound_score > -0.5) and (compound_score < 0.5):
            rating = values[1]
        elif compound_score <= -0.5:
            rating = values[2]
        sent_dict = {'Positive': ':green[**Positive**]', 'Negative': ':red[**Negative**]', 'Neutral': '**Neutral**'}
        return rating, sent_dict
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return 'Neutral', {'Neutral': '**Neutral**'}

@st.cache_data(show_spinner=False)        
def topic(article):
    '''
    Topic modelling for the article using Latent Dirichlet Allocation 
    '''
    try:
        text = [preprocess(article)]
        count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
        x_counts = count_vect.fit_transform(text)
        x_counts.todense()  
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_counts)
        dimension = 1
        lda = LDA(n_components=dimension)
        lda_array = lda.fit_transform(x_tfidf)
        components = [lda.components_[i] for i in range(len(lda.components_))]
        features = list(count_vect.get_feature_names_out())
        important_words = [sorted(features, key=lambda x: components[j][features.index(x)], reverse=True)[:3] for j in range(len(components))]
        words = ''
        c = 0
        for i in important_words:
            for y in i:
                c += 1
                if c == 1:
                    words += y + ', '
                elif c == 2:
                    words += y + ' and '
                else:
                    words += y
        return words
    except Exception as e:
        st.error(f"Error in topic modeling: {str(e)}")
        return "general news"

# MAIN APP CONTENT
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
    
    # URL detection pattern
    pattern = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»""''])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))')
    
    matches = pattern.findall(text)
    if len(matches) == 1: #if link found in input it will go to llm to get article contents
        scraped_text = scrape(text)
        if scraped_text:
            text = scraped_text
        else:
            st.error("Failed to scrape the article. Please try pasting the article text directly.")
            st.stop()
    
    # Only check database if available
    verify = check_db(text) if DB_AVAILABLE else False
    
    if verify == False:
        stuff = predict(text)
        st.markdown('*please wait while our Large Language Model checks the facts of this article...*')
        
        # Add delay to prevent rate limiting and use simplified agent
        time.sleep(2)
        try:
            result = agent(text)
        except Exception as e:
            if "ratelimit" in str(e).lower():
                st.warning("⏰ Rate limit reached. Please wait a few minutes before trying again.")
                result = "Rate limit reached - please try again in a few minutes."
            else:
                st.write('An Error has occurred please try again later!')
                result = 'NA' 
        
        st.markdown('**Disclaimer**⚠️ Machine Learning is not 100 percent accurate and can make mistakes')
        sentiment, sentiment_coloured = get_sentiment(text)
        text_list = [text]
        topics = topic(text_list)    
        st.markdown(f'Additionally we found that this news article with the keywords of "{topics}" has a {sentiment_coloured[sentiment]} sentiment')
        
        # Save to database
        if DB_AVAILABLE:
            save_success = save_to_db(text, stuff[1], result, sentiment, topics)
            if save_success:
                st.success("✅ Results saved to database!")
            else:
                st.info("ℹ️ Results displayed but database save failed.")
        else:
            st.info("ℹ️ Results displayed but not saved (database not available).")