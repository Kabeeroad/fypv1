import streamlit as st
import sqlite3
import pandas as pd
from datetime import date

st.set_page_config(page_title="About", page_icon="🌎")

st.markdown("# About")
st.write("This is a webapp that has implemented cutting edge tools from our research aiming to tackle fake news by utilising Machine Learning and Large Language models.")
#st.write("See the research paper here -> TBC")

# Database statistics section
DB_PATH = 'fake_news_data.db'

def get_db_stats():
    """Get database statistics for the about page"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
        if not cursor.fetchone():
            return None
        
        # Get total articles analyzed
        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]
        
        # Get fake vs real breakdown
        cursor.execute("SELECT classification, COUNT(*) FROM articles GROUP BY classification")
        breakdown = dict(cursor.fetchall())
        
        # Get most recent analysis date
        cursor.execute("SELECT MAX(date) FROM articles")
        last_analysis = cursor.fetchone()[0]
        
        conn.close()
        return {
            'total': total,
            'breakdown': breakdown,
            'last_analysis': last_analysis
        }
    except Exception:
        return None

# Show database stats if available
stats = get_db_stats()
if stats and stats['total'] > 0:
    st.markdown("---")
    st.subheader("📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles Analyzed", stats['total'])
    with col2:
        fake_count = stats['breakdown'].get('Fake', 0)
        st.metric("Fake News Detected", fake_count)
    with col3:
        real_count = stats['breakdown'].get('Real', 0)
        st.metric("Real News Verified", real_count)
    
    if stats['last_analysis']:
        st.info(f"Last analysis performed: {stats['last_analysis']}")

tab1, tab2, = st.tabs(["Purpose", "Technical Specifications"])

with tab1:
    st.subheader("Purpose")
    st.write("Fake News is becoming so prevalent in our societies and has been proven to influence political views and even cause harm to individuals and organisations. This situation has been escalated since Large Language Models (e.g.ChatGPT) can massively produce fake news written in the style of fake news. We need to ensure what we read online is true.")
    st.write("This Fake News Detector is a frontline response in classifying if articles are false or real!")
    
    st.markdown("### Why This Matters")
    st.write("""
    - **Information Integrity**: Ensuring accurate information reaches the public
    - **Democratic Process**: Protecting elections and civic discourse from misinformation
    - **Public Health**: Preventing spread of medical misinformation
    - **Social Harmony**: Reducing conflict caused by false narratives
    - **Media Literacy**: Educating users about information verification
    """)

with tab2:
    st.subheader("Technical Specifications")
    st.write("The fake news detector framework works like this:")
    
    # Check if the framework image exists
    try:
        st.image("pages/images/Framework.png", caption="Fake News Detection Framework")
    except:
        # If image doesn't exist, show text-based framework
        st.markdown("""
        ### Detection Framework:
        
        **Step 1: Input Processing**
        - Article text extraction (from URL or direct input)
        - Text preprocessing and normalization
        
        **Step 2: Machine Learning Classification**
        - Logistic Regression model trained on 76,000+ articles
        - TF-IDF vectorization for feature extraction
        - 95% accuracy on test dataset
        
        **Step 3: LLM Fact-Checking**
        - Meta's Llama3-70B model for claim verification
        - DuckDuckGo and Wikipedia search integration
        - Cross-referencing with reliable sources
        
        **Step 4: Analysis & Results**
        - Sentiment analysis using VADER
        - Topic modeling with Latent Dirichlet Allocation
        - Comprehensive report generation
        """)
    
    st.write("This uses a Logistic Regression algorithim trained on around 76,000 news articles achieving 86.95 accuracy.")
    st.write("To fact check the claims, Meta's llama3-70b LLM model with the tools of Duckduckgo and Wikipedia are used.")
    
    # st.markdown("### Technology Stack")
    col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Machine Learning:**
#         - Scikit-learn (Logistic Regression)
#         - NLTK (Natural Language Processing)
#         - TF-IDF Vectorization
#         - Pandas & NumPy
#         """)
    
#     with col2:
#         st.markdown("""
#         **AI & Search:**
#         - LangChain Framework
#         - Groq API (Llama3-70B)
#         - DuckDuckGo Search API
#         - Wikipedia API
#         """)
    
#     st.markdown("""
#     **Database & Interface:**
#     - SQLite Database
#     - Streamlit Web Framework
#     - Matplotlib & WordCloud Visualization
#     """)

# with tab3:
#     st.subheader("How It Works")
    
#     st.markdown("### 🔍 Detection Process")
    
#     st.markdown("""
#     **1. Article Input**
#     - Paste article text directly, or
#     - Provide a URL for automatic content extraction
    
#     **2. Preprocessing**
#     - Remove special characters and URLs
#     - Tokenization and lemmatization
#     - Stop word removal
#     - Text normalization
    
#     **3. ML Classification**
#     - Convert text to TF-IDF vectors
#     - Apply trained Logistic Regression model
#     - Generate probability scores for Fake/Real
    
#     **4. LLM Fact-Checking**
#     - Extract key claims from the article
#     - Search Wikipedia and web sources
#     - Verify facts against reliable sources
#     - Generate detailed explanation
    
#     **5. Additional Analysis**
#     - Sentiment analysis (Positive/Negative/Neutral)
#     - Topic modeling to identify key themes
#     - Confidence scoring and explanations
#     """)
    
    st.markdown("### 💡 Best Practices")
    st.info("""
    **For Best Results:**
    - Provide complete article text (not just headlines)
    - Avoid including advertisements or unrelated content
    - Use recent articles for better fact-checking accuracy
    - Cross-reference results with multiple sources when in doubt
    """)
    
    st.warning("""
    **Limitations:**
    - Machine learning models are not 100% accurate
    - Fact-checking depends on available online sources
    - New or very recent events may not be verifiable
    - Always use critical thinking and multiple sources
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<small>
Built with ❤️ using Streamlit, powered by AI and Machine Learning<br>
For research and educational purposes
</small>
</div>
""", unsafe_allow_html=True)