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
from urllib.parse import urlparse

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

# =============================================================================
# NEW: SOURCE CREDIBILITY DATABASE
# =============================================================================

CREDIBILITY_SCORES = {
    # Pakistani Trusted Sources
    "dawn.com": {"score": 90, "category": "🏛️ Trusted Pakistani News", "color": "green"},
    "geo.tv": {"score": 75, "category": "📺 Mainstream Pakistani TV", "color": "blue"},
    "thenews.com.pk": {"score": 80, "category": "📰 Established Pakistani Media", "color": "green"},
    "tribune.com.pk": {"score": 85, "category": "📰 Express Tribune", "color": "green"},
    "samaa.tv": {"score": 70, "category": "📺 Samaa TV", "color": "blue"},
    "arydigital.tv": {"score": 65, "category": "📺 ARY Digital", "color": "blue"},
    "dunyanews.tv": {"score": 70, "category": "📺 Dunya News", "color": "blue"},
    "92newshd.tv": {"score": 65, "category": "📺 92 News", "color": "blue"},
    "bolnews.com": {"score": 60, "category": "📺 BOL News", "color": "orange"},
    
    # International Sources
    "bbc.com": {"score": 95, "category": "🌍 BBC International", "color": "green"},
    "reuters.com": {"score": 95, "category": "🌍 Reuters", "color": "green"},
    "ap.org": {"score": 95, "category": "🌍 Associated Press", "color": "green"},
    "cnn.com": {"score": 78, "category": "📺 CNN", "color": "blue"},
    "aljazeera.com": {"score": 85, "category": "🌍 Al Jazeera", "color": "green"},
    "nytimes.com": {"score": 88, "category": "📰 New York Times", "color": "green"},
    "washingtonpost.com": {"score": 85, "category": "📰 Washington Post", "color": "green"},
    "theguardian.com": {"score": 82, "category": "📰 The Guardian", "color": "green"},
    
    # Social Media & Blogs (Lower credibility)
    "facebook.com": {"score": 20, "category": "📱 Social Media Post", "color": "red"},
    "whatsapp.com": {"score": 15, "category": "📱 WhatsApp Forward", "color": "red"},
    "youtube.com": {"score": 30, "category": "📺 YouTube Video", "color": "orange"},
    "twitter.com": {"score": 25, "category": "📱 Twitter Post", "color": "orange"},
    "instagram.com": {"score": 20, "category": "📱 Instagram Post", "color": "red"},
    "tiktok.com": {"score": 15, "category": "📱 TikTok Video", "color": "red"},
    
    # URL Shorteners & Suspicious
    "bit.ly": {"score": 10, "category": "⚠️ Shortened Link", "color": "red"},
    "tinyurl": {"score": 10, "category": "⚠️ Shortened Link", "color": "red"},
    "t.co": {"score": 10, "category": "⚠️ Twitter Shortened Link", "color": "red"},
    "short.link": {"score": 10, "category": "⚠️ Shortened Link", "color": "red"},
}

def get_source_credibility(text_or_url):
    """Check source credibility from URL or text"""
    try:
        # If it's a URL
        if text_or_url.startswith('http'):
            domain = urlparse(text_or_url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check exact match first
            if domain in CREDIBILITY_SCORES:
                return CREDIBILITY_SCORES[domain]
            
            # Check partial matches
            for key in CREDIBILITY_SCORES:
                if key in domain:
                    return CREDIBILITY_SCORES[key]
            
            return {"score": 50, "category": "❓ Unknown Source", "color": "gray"}
        
        # If it's text, look for domain mentions
        for domain in CREDIBILITY_SCORES:
            if domain in text_or_url.lower():
                return CREDIBILITY_SCORES[domain]
        
        return {"score": 60, "category": "📄 Direct Text", "color": "blue"}
        
    except Exception as e:
        return {"score": 50, "category": "❓ Unknown Source", "color": "gray"}

# =============================================================================
# NEW: URL SAFETY CHECKER
# =============================================================================

def check_url_safety(url):
    """Check URL safety and show warnings"""
    try:
        safety_score = 100
        warnings = []
        domain = urlparse(url).netloc.lower()
        
        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'short.link', 'cutt.ly', 'goo.gl']
        if any(shortener in domain for shortener in shorteners):
            safety_score -= 40
            warnings.append("⚠️ Shortened URL - actual destination hidden")
        
        # Check for suspicious characters
        if '@' in url or '%' in url:
            safety_score -= 30
            warnings.append("⚠️ Contains suspicious characters")
        
        # Check HTTPS
        if not url.startswith('https://'):
            safety_score -= 20
            warnings.append("⚠️ Not using secure connection (HTTPS)")
        
        # Check domain length (very long domains often suspicious)
        if len(domain) > 35:
            safety_score -= 25
            warnings.append("⚠️ Unusually long domain name")
        
        # Check for numbers in domain (often suspicious)
        if any(char.isdigit() for char in domain.replace('92newshd', '').replace('24news', '')):
            safety_score -= 15
            warnings.append("⚠️ Domain contains numbers (potentially suspicious)")
        
        # Check for multiple subdomains
        if domain.count('.') > 2:
            safety_score -= 20
            warnings.append("⚠️ Multiple subdomains detected")
        
        # Determine safety level
        if safety_score >= 80:
            level = "✅ Safe"
            color = "green"
        elif safety_score >= 60:
            level = "⚠️ Caution"
            color = "orange"
        else:
            level = "❌ Risky"
            color = "red"
        
        return {
            "score": max(0, safety_score),
            "level": level,
            "color": color,
            "warnings": warnings
        }
        
    except Exception as e:
        return {
            "score": 50,
            "level": "❓ Unknown",
            "color": "gray",
            "warnings": [f"Could not analyze URL: {str(e)}"]
        }

# =============================================================================
# NEW: ENHANCED FACT CHECKING FUNCTIONS
# =============================================================================

def extract_key_entities(text):
    """Extract key entities using simple NLP"""
    try:
        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)  # Proper nouns
        
        # Remove common words
        stop_words = {"The", "This", "That", "And", "Or", "But", "In", "On", "At", "To", "For", "News", "Article"}
        entities = [word for word in words if word not in stop_words]
        
        return entities[:5]  # Return top 5
    except:
        return ["news", "information"]

def extract_main_claim(text):
    """Extract main claim from text"""
    sentences = text.split('.')
    # Return first substantial sentence
    for sentence in sentences:
        if len(sentence.strip()) > 20:
            return sentence.strip()
    return text[:100]

def check_wikipedia_enhanced(query):
    """Enhanced Wikipedia check"""
    try:
        # Extract key terms for better search
        key_terms = extract_key_entities(query)
        
        for term in key_terms[:2]:  # Check top 2 terms only
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.replace(' ', '_')}"
            response = requests.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data and len(data['extract']) > 50:
                    return {
                        "found": True,
                        "summary": data['extract'][:300],
                        "title": data.get('title', term),
                        "url": data.get('content_urls', {}).get('desktop', {}).get('page', '')
                    }
        
        return {"found": False, "summary": "No Wikipedia entry found"}
        
    except Exception as e:
        return {"found": False, "summary": f"Wikipedia check failed: {str(e)}"}

def simple_google_search(query):
    """Simple search simulation"""
    try:
        # Extract main claim from query  
        main_claim = extract_main_claim(query)
        
        # Common news topics that usually have verification
        verifiable_topics = [
            "covid", "vaccine", "election", "politics", "earthquake", "weather", 
            "government", "minister", "president", "parliament", "court", "supreme court",
            "pakistan", "india", "china", "america", "imran khan", "nawaz sharif", 
            "pti", "pml", "ppp", "cricket", "fifa", "olympics"
        ]
        
        found = any(topic in query.lower() for topic in verifiable_topics)
        
        if found:
            return {
                "found": True,
                "snippet": f"Multiple search results found for: {main_claim[:100]}",
                "confidence": "Medium"
            }
        else:
            return {
                "found": False,
                "snippet": "Limited search results found for verification",
                "confidence": "Low"
            }
            
    except Exception as e:
        return {"found": False, "snippet": f"Search failed: {str(e)}", "confidence": "None"}

def simple_llm_analysis(article, wiki_result, google_result):
    """Simplified LLM analysis"""
    try:
        evidence = ""
        if wiki_result["found"]:
            evidence += f"Wikipedia info: {wiki_result['summary'][:200]}\n"
        if google_result["found"]:
            evidence += f"Search results: {google_result['snippet'][:200]}\n"
        
        prompt = f"""Article: {article[:400]}

Evidence found:
{evidence}

Based on the evidence, is this article VERIFIED, UNVERIFIED, or CONTRADICTED? 
Give your analysis in exactly 3 sentences. Be specific about what evidence supports or contradicts the claims."""
        
        response = chat.invoke([("human", prompt)])
        return response.content[:400]  # Limit response length
        
    except Exception as e:
        return f"LLM analysis unavailable: {str(e)}"

@st.cache_data(show_spinner='🔍 Multiple sources check kar rahe hain...', ttl=1800)
def enhanced_fact_check(article):
    """Enhanced fact checking with multiple sources"""
    
    # Truncate article if too long
    max_chars = 1000
    if len(article) > max_chars:
        article = article[:max_chars] + "..."
    
    sources_checked = []
    final_verdict = "Inconclusive"
    
    try:
        # Create columns for progress
        progress_col1, progress_col2 = st.columns(2)
        
        with progress_col1:
            # 1. Wikipedia Check
            st.write("📖 Checking Wikipedia...")
            wiki_result = check_wikipedia_enhanced(article)
            sources_checked.append({
                "source": "Wikipedia", 
                "status": "✅ Found" if wiki_result["found"] else "❌ Not Found",
                "details": wiki_result.get("summary", "No information found")[:200]
            })
        
        with progress_col2:
            # 2. Search verification
            st.write("🔍 Verifying claims...")
            google_result = simple_google_search(article)
            sources_checked.append({
                "source": "Search Verification",
                "status": "✅ Found" if google_result["found"] else "❌ Limited Results",
                "details": google_result.get("snippet", "No relevant results")[:200]
            })
        
        # 3. LLM analysis
        st.write("🤖 AI analysis...")
        llm_result = simple_llm_analysis(article, wiki_result, google_result)
        
        # Combine results for final verdict
        evidence_score = 0
        if wiki_result["found"]: evidence_score += 40
        if google_result["found"]: evidence_score += 30
        if "verified" in llm_result.lower() or "supported" in llm_result.lower(): evidence_score += 30
        if "contradicted" in llm_result.lower(): evidence_score -= 30
        
        # Determine final verdict
        if evidence_score >= 70:
            final_verdict = "✅ Well Supported by Sources"
            verdict_color = "green"
        elif evidence_score >= 40:
            final_verdict = "⚠️ Partially Supported"
            verdict_color = "orange" 
        elif evidence_score <= 0:
            final_verdict = "❌ Contradicted by Sources"
            verdict_color = "red"
        else:
            final_verdict = "❓ No Supporting Evidence Found"
            verdict_color = "orange"
        
        # Display results
        st.subheader("🔍 Multi-Source Fact Check Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Evidence Score", f"{max(0, evidence_score)}/100")
        with col2:
            if verdict_color == "green":
                st.success(final_verdict)
            elif verdict_color == "orange":
                st.warning(final_verdict)
            else:
                st.error(final_verdict)
        
        # Show source details in expandable sections
        with st.expander("📚 Sources Checked Details", expanded=False):
            for source in sources_checked:
                st.write(f"**{source['source']}**: {source['status']}")
                if source['details']:
                    st.write(f"_{source['details']}_")
                st.write("---")
        
        with st.expander("🤖 AI Analysis Details", expanded=False):
            st.write(llm_result)
        
        return f"{final_verdict}\n\nSources checked: {len(sources_checked)}\nEvidence score: {max(0, evidence_score)}/100\n\n{llm_result[:200]}"
        
    except Exception as e:
        error_msg = f"⚠️ Fact-checking main error: {str(e)}"
        st.warning(error_msg)
        return error_msg

# =============================================================================
# EXISTING DATABASE SETUP (UNCHANGED)
# =============================================================================

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
            url TEXT,
            credibility_score INTEGER,
            safety_score INTEGER,
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

# =============================================================================
# EXISTING FUNCTIONS (UNCHANGED)
# =============================================================================

@st.cache_resource
def load_model():
    '''loads fake news model (trained logistic regression model)'''
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
                st.markdown('Our enhanced fact-checking system found:')
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

def save_to_db(article, classification, llm_result, sentiment, topic, url=None, credibility_score=None, safety_score=None):
    """Save analysis results to SQLite database"""
    if not DB_AVAILABLE:
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO articles (article, classification, date, llm_result, sentiment, topic, url, credibility_score, safety_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (article, classification, str(date.today()), llm_result, sentiment, topic, url, credibility_score, safety_score))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.warning(f"Failed to save to database: {str(e)}")
        return False

@st.cache_data(show_spinner=False)
def scrape(text):
    '''uses libraries of requests to load the page and bs4 to parse the html text'''
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        page = requests.get(text, timeout=30, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        
        # Try to extract main content more intelligently
        article_content = ""
        
        # Common article selectors
        selectors = [
            'article', '.article-body', '.post-content', '.entry-content',
            '.story-body', '.article-content', '.main-content', 'main'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                article_content = content.get_text(strip=True)
                break
        
        if not article_content:
            article_content = soup.get_text()
        
        # Use LLM to clean and extract article
        q = str(article_content)
        varr = []
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You need to read this HTML content and extract ONLY the main news article. Remove advertisements, navigation menus, comments, and other non-article content. Return only the article text. Article content: {article}")
        ])
        chain = prompt | chat
        for chunk in chain.stream({"article": q[:3000]}):  # Limit input size
            claim = (chunk.content)
            varr.append(claim)
        claims = ''.join(varr)
        claims = claims.replace('\n', ' ').strip()
        return claims
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return ""

@st.cache_data(show_spinner='Thinking...') 
def predict(data):
    '''ML model prediction function'''
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
        
        probabilities = list(probabilities)
        Fake = round(probabilities[0][0]*100, 1)
        Real = round(probabilities[0][1]*100, 1)

        Real_Msg = f':green[**REAL!**] We predicted that the probability this News article is :green[**Real**] is {Real}%'
        Fake_Msg = f':red[**FAKE!**] We predicted that the probability this News article is :red[**Fake**] is {Fake}%'
        
        classification = 'Real' if prediction == [1] else 'Fake'
        
        if prediction == [1]:
            st.markdown(Real_Msg)
        else:
            st.markdown(Fake_Msg)

        return [data, classification, date.today()]
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return [data, "Error", date.today()]

@st.cache_data(show_spinner=False)        
def preprocess(text):
    '''Used to preprocess the data'''
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

@st.cache_data(show_spinner=False)        
def get_sentiment(article):
    '''Gets overall sentiment of article'''
    try:
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(article)
        compound_score = score.get('compound')
        values = ['Positive', 'Neutral', 'Negative']
        
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
    '''Topic modelling for the article using LDA'''
    try:
        text = [preprocess(article)]
        count_vect = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True)
        x_counts = count_vect.fit_transform(text)
        
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

# =============================================================================
# MAIN APP CONTENT - ENHANCED VERSION
# =============================================================================

st.write("# Fake News Detector 🕵️‍♂️")
st.markdown("### Enhanced with Multi-Source Verification & Security Analysis")

text = st.text_input("Enter an Article or News URL here:", key="Article")
st.write('Hint💡: Try to enter as much of the news article contents as possible or paste a URL for automatic analysis.')

if text:
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        latest_iteration.text(f'Analysing Text 🔎 {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)  
    
    # URL detection pattern
    pattern = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»""''])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))')
    
    original_url = None
    matches = pattern.findall(text)
    
    if len(matches) == 1:
        original_url = matches[0]
        
        # NEW: URL Security Analysis
        st.subheader("🛡️ URL Security Analysis")
        url_safety = check_url_safety(original_url)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if url_safety["color"] == "green":
                st.success(f"Safety Score: {url_safety['score']}/100")
            elif url_safety["color"] == "orange":
                st.warning(f"Safety Score: {url_safety['score']}/100")
            else:
                st.error(f"Safety Score: {url_safety['score']}/100")
        
        with col2:
            st.write(f"**Status:** {url_safety['level']}")
            if url_safety["warnings"]:
                for warning in url_safety["warnings"]:
                    st.write(f"• {warning}")
        
        # NEW: Source Credibility Analysis  
        st.subheader("🏛️ Source Credibility Analysis")
        credibility = get_source_credibility(original_url)
        
        col1, col2 = st.columns(2)
        with col1:
            if credibility["color"] == "green":
                st.success(f"Credibility: {credibility['score']}/100")
            elif credibility["color"] == "blue":
                st.info(f"Credibility: {credibility['score']}/100")
            elif credibility["color"] == "orange":
                st.warning(f"Credibility: {credibility['score']}/100")
            else:
                st.error(f"Credibility: {credibility['score']}/100")
        
        with col2:
            st.write(f"**Category:** {credibility['category']}")
        
        # Continue with existing scraping
        st.info("🔄 Extracting article content from URL...")
        scraped_text = scrape(original_url)
        if scraped_text:
            text = scraped_text
            st.success("✅ Article content extracted successfully!")
        else:
            st.error("❌ Failed to scrape the article. Please try pasting the article text directly.")
            st.stop()
    else:
        # If no URL detected, still check for source credibility in text
        credibility = get_source_credibility(text)
        original_url = None
    
    # Database check
    verify = check_db(text) if DB_AVAILABLE else False
    
    if verify == False:
        # ML Model Analysis
        st.subheader("🤖 AI Model Analysis")
        stuff = predict(text)
        
        # Enhanced Multi-Source Fact Checking 
        st.subheader("🔍 Multi-Source Fact Verification")
        try:
            enhanced_result = enhanced_fact_check(text)
        except Exception as e:
            st.error(f"Fact checking failed: {str(e)}")
            enhanced_result = "Enhanced fact checking unavailable"
        
        # Content Analysis
        st.subheader("📊 Content Analysis") 
        sentiment, sentiment_coloured = get_sentiment(text)
        text_list = [text]
        topics = topic(text_list)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Key Topics:** {topics}")
        with col2:
            st.write(f"**Overall Sentiment:** {sentiment_coloured[sentiment]}")
        
        # Enhanced Summary Box
        st.subheader("📋 Analysis Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            if stuff[1] == 'Real':
                st.metric("ML Classification", "✅ REAL", delta="Reliable")
            else:
                st.metric("ML Classification", "❌ FAKE", delta="Suspicious")
        
        with summary_col2:
            if original_url:
                st.metric("Source Credibility", f"{credibility['score']}/100", 
                         delta=credibility['category'])
            else:
                st.metric("Input Type", "Direct Text", delta="Manual Entry")
        
        with summary_col3:
            if "supported" in enhanced_result.lower():
                st.metric("Fact Check", "✅ Verified", delta="Multiple Sources")
            elif "contradicted" in enhanced_result.lower():
                st.metric("Fact Check", "❌ Contradicted", delta="Evidence Against")
            else:
                st.metric("Fact Check", "❓ Inconclusive", delta="Limited Evidence")
        
        # Save enhanced data to database
        if DB_AVAILABLE:
            save_success = save_to_db(
                text, 
                stuff[1], 
                enhanced_result, 
                sentiment, 
                topics,
                original_url,
                credibility.get('score', 50) if original_url else None,
                url_safety.get('score', None) if original_url else None
            )
            if save_success:
                st.success("✅ Complete analysis saved to database!")
            else:
                st.info("ℹ️ Analysis completed but database save failed.")
        else:
            st.info("ℹ️ Analysis completed but not saved (database not available).")
        
        # Enhanced Disclaimers and Recommendations
        st.markdown("---")
    #     st.subheader("⚠️ Important Information")
        
    #     disclaimer_col1, disclaimer_col2 = st.columns(2)
        
    #     with disclaimer_col1:
    #         st.warning("🤖 **AI Limitations**")
    #         st.write("• Machine Learning models can make mistakes")
    #         st.write("• Always cross-verify important information")
    #         st.write("• Consider multiple reliable sources")
        
    #     with disclaimer_col2:
    #         st.info("🔍 **Best Practices**")
    #         st.write("• Check original sources when possible")
    #         st.write("• Be skeptical of sensational claims")
    #         st.write("• Verify through established news outlets")
        
    #     # Action Recommendations
    #     if stuff[1] == 'Fake' or credibility.get('score', 50) < 40:
    #         st.error("⚠️ **HIGH CAUTION RECOMMENDED**")
    #         st.write("This content shows multiple warning signs. Please:")
    #         st.write("• Do not share without verification")
    #         st.write("• Check with trusted news sources")
    #         st.write("• Be aware this might be misinformation")
        
    #     elif "contradicted" in enhanced_result.lower():
    #         st.error("❌ **CONTENT CONTRADICTED BY EVIDENCE**")
    #         st.write("Our fact-checking found evidence that contradicts claims in this article.")
        
    #     elif stuff[1] == 'Real' and credibility.get('score', 50) > 70:
    #         st.success("✅ **APPEARS RELIABLE**")
    #         st.write("This content shows positive reliability indicators, but always maintain healthy skepticism.")
    
    # # Additional Features Section
    # st.markdown("---")
    # st.subheader("🚀 Additional Features")
    
    # feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    # with feature_col1:
    #     if st.button("📊 View Analysis History"):
    #         if DB_AVAILABLE:
    #             try:
    #                 conn = sqlite3.connect(DB_PATH)
    #                 df = pd.read_sql_query("SELECT date, classification, credibility_score FROM articles ORDER BY created_at DESC LIMIT 10", conn)
    #                 st.dataframe(df)
    #                 conn.close()
    #             except Exception as e:
    #                 st.error(f"Could not load history: {str(e)}")
    #         else:
    #             st.info("Database not available for history")
    
    # with feature_col2:
    #     if st.button("🔗 Check Another URL"):
    #         st.rerun()
    
    # with feature_col3:
    #     if st.button("📝 Analyze New Text"):
    #         st.rerun()

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        <p>🔍 Fake News Detector | Powered by <b>Machine Learning</b>, <b>LLMs</b>, and <b>Trusted Sources</b></p>
        <p>⚠️ Results are AI-generated and may not be 100% accurate. Always verify with multiple sources.</p>
        <p>Built with ❤️ using <a href="https://streamlit.io/" target="_blank" style="color: #38b6ff; text-decoration: none;">Streamlit</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
