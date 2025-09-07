import streamlit as st
import time
import pandas as pd
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex
import nltk
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import date, timedelta

st.title('Analytics📈')

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

# Database connection with error handling
DB_AVAILABLE = False

try:
    DB_AVAILABLE = init_database()
    st.success("✅ Database connected successfully!")
except Exception as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Database connection failed: {str(e)}")

def get_article_counts():
    """Get total and fake article counts from database"""
    if not DB_AVAILABLE:
        return 0, 0
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]
        
        # Get fake count
        cursor.execute("SELECT COUNT(*) FROM articles WHERE LOWER(classification) = 'fake'")
        fake = cursor.fetchone()[0]
        
        conn.close()
        return total, fake
    except Exception as e:
        st.error(f"Error getting counts: {str(e)}")
        return 0, 0

def get_articles_by_timeframe(days=None):
    """Get articles filtered by time period"""
    if not DB_AVAILABLE:
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(DB_PATH)
        
        if days is None:
            # Get all articles
            query = "SELECT * FROM articles"
            df = pd.read_sql_query(query, conn)
        else:
            # Get articles from specific time period
            cutoff_date = (date.today() - timedelta(days=days)).strftime('%Y-%m-%d')
            query = "SELECT * FROM articles WHERE date >= ?"
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error getting articles: {str(e)}")
        return pd.DataFrame()

def generate_sample_data():
    """Generate sample data when no real data is available"""
    import random
    
    sample_data = []
    for i in range(50):
        classification = random.choice(['Fake', 'Real'])
        sentiment = random.choice(['Positive', 'Negative', 'Neutral'])
        article_date = date.today() - timedelta(days=random.randint(1, 365))
        topic = random.choice(['politics', 'health', 'technology', 'sports', 'entertainment'])
        
        sample_data.append({
            'classification': classification,
            'date': article_date.strftime('%Y-%m-%d'),
            'sentiment': sentiment,
            'topic': topic,
            'article': f'Sample {classification.lower()} news article about {topic}'
        })
    
    return pd.DataFrame(sample_data)

# Main analytics logic
if not DB_AVAILABLE:
    st.warning("Analytics features require database connection. Using sample data for demonstration.")
    df_sample = generate_sample_data()
    total = len(df_sample)
    fake = len(df_sample[df_sample['classification'] == 'Fake'])
else:
    total, fake = get_article_counts()
    
    if total == 0:
        st.info("No data found in database. Using sample data for demonstration.")
        df_sample = generate_sample_data()
        total = len(df_sample)
        fake = len(df_sample[df_sample['classification'] == 'Fake'])

real = total - fake

st.write(f'This fake news detector has identified {fake} out of {total} articles as being fake news!')

option = st.selectbox(
    "See how many Fake and Real articles our detector has classified over different time periods",
    ("Last 7 Days", "Last 14 Days", "Last Month", "Last 6 Months", "Last Year", "All Time"))

def create_classification_chart(data, title):
    """Create a bar chart for classifications"""
    if data.empty:
        st.warning(f"No data available for {title}")
        return
        
    counts = data['classification'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = {'Fake': '#ff4c4c', 'Real': '#38d864'}
    
    bars = ax.bar(counts.index, counts.values)
    for bar, classification in zip(bars, counts.index):
        bar.set_color(colours.get(classification, '#cccccc'))
    
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none')
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    ax.set_title(f"Classifications - {title}")
    st.pyplot(fig)

# Time period mapping
time_periods = {
    'Last 7 Days': 7,
    'Last 14 Days': 14,
    'Last Month': 30,
    'Last 6 Months': 180,
    'Last Year': 365,
    'All Time': None
}

try:
    if DB_AVAILABLE and total > 0:  # Real data exists
        days = time_periods[option]
        data = get_articles_by_timeframe(days)
    else:  # Use sample data
        data = df_sample
        # Filter sample data based on option
        if option != 'All Time':
            days = time_periods[option]
            cutoff_date = date.today() - timedelta(days=days)
            data['date'] = pd.to_datetime(data['date'])
            data = data[data['date'] >= pd.to_datetime(cutoff_date)]
    
    create_classification_chart(data, option)
    
except Exception as e:
    st.error(f"Error querying data for {option}: {str(e)}")

# Word cloud section
try:
    if DB_AVAILABLE and total > 0:  # Real data
        df_all = get_articles_by_timeframe()
    else:  # Sample data
        df_all = df_sample
    
    fake_articles = df_all[df_all['classification'] == 'Fake']['article'] if 'article' in df_all.columns else []
    real_articles = df_all[df_all['classification'] == 'Real']['article'] if 'article' in df_all.columns else []

    if len(fake_articles) > 0:
        st.header('Most common words in Fake News articles:')
        text = " ".join(str(t) for t in fake_articles if pd.notna(t))
        if text.strip():
            word_cloud = WordCloud(collocations=False, background_color='black').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(word_cloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No fake article text available for word cloud.")

    if len(real_articles) > 0:
        st.header('Most common words in Real News articles:')
        text = " ".join(str(t) for t in real_articles if pd.notna(t))
        if text.strip():
            word_cloud = WordCloud(collocations=False, background_color='black').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(word_cloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No real article text available for word cloud.")

except Exception as e:
    st.error(f"Error creating word clouds: {str(e)}")

# Sentiment Analysis
st.header('Sentiment Analysis:')
sentiment_option = st.selectbox(
    "Find out the Sentiments of the Articles that our model has classified",
    ("All Articles", "Fake Articles", "Real Articles"))

def create_sentiment_pie_chart(data, title):
    """Create a pie chart for sentiment analysis"""
    if data.empty or 'sentiment' not in data.columns:
        st.warning(f"No sentiment data available for {title}")
        return
        
    counts = data['sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(11, 6))
    colours = {'Negative': '#ff4c4c', 'Positive': '#38d864', 'Neutral': '#ffff00'}
    colors = [colours.get(sentiment, '#cccccc') for sentiment in counts.index]
    
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, textprops={'color': "white"})
    ax.axis('equal')
    ax.set_title(f"Sentiment Analysis - {title}", color='white')
    st.pyplot(fig)

try:
    if sentiment_option == 'All Articles':
        sentiment_data = df_all
    elif sentiment_option == 'Fake Articles':
        sentiment_data = df_all[df_all['classification'] == 'Fake'] if not df_all.empty else pd.DataFrame()
    else:  # Real Articles
        sentiment_data = df_all[df_all['classification'] == 'Real'] if not df_all.empty else pd.DataFrame()
    
    create_sentiment_pie_chart(sentiment_data, sentiment_option)
    
except Exception as e:
    st.error(f"Error creating sentiment analysis: {str(e)}")

# Database statistics
if DB_AVAILABLE:
    st.header('Database Statistics:')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", total)
    with col2:
        st.metric("Fake Articles", fake)
    with col3:
        st.metric("Real Articles", real)
    
    if total > 0:
        accuracy_display = f"{((real / total) * 100):.1f}%"
        st.metric("Real Article Rate", accuracy_display)