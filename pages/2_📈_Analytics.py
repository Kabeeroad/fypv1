import streamlit as st
import time
import pandas as pd
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex
import nltk
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title('Analytics📈')

conn = st.connection("gsheets",type=GSheetsConnection)
sqlquery="""
SELECT COUNT(Classification) FROM Sheet1 where Classification == 'Fake'
""" #uses sql select statement to get updated result
select=conn.query(sql=sqlquery,ttl=20)
df=pd.DataFrame(select)
fake = int(df['count(Classification)'].iloc[0])

sql="""
SELECT COUNT(Classification) FROM Sheet1 
""" 
select=conn.query(sql=sql,ttl=20)
df=pd.DataFrame(select)
real = int(df['count(Classification)'].iloc[0])
st.write(f'This fake news detector has identified {fake} out of {real} articles as being fake news!')

option = st.selectbox(
    "See how many Fake and Real articles our detector has classified over different time periods",
    ("Last 7 Days", "Last 14 Days", "Last Month", "Last 6 Months", "Last Year", "All Time"))

if option == 'Last 7 Days':
    sevenDays = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 7 
    """
    #cast converts date column from sheets from str to date
    select=conn.query(sql=sevenDays,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  

elif option == 'Last 14 Days':
    fourtheenDays = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 14 
    """
    select=conn.query(sql=fourtheenDays,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  

elif option == 'Last Month':
    oneMonth = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 30 
    """
    select=conn.query(sql=oneMonth,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  

elif option == 'Last 6 Months':
    sixMonths = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 180 
    """
    select=conn.query(sql=sixMonths,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  

elif option == 'Last Year':
    oneYear = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 356 
    """
    select=conn.query(sql=oneYear,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  

elif option == 'All Time':
    total = 'SELECT* FROM Sheet1'
    select=conn.query(sql=total,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    fig, ax = plt.subplots(figsize = (10, 5))
    colours = {'Fake':'#ff4c4c','Real':'#38d864'}
    value_colours = [colours[x] for x in counts.index] 
    bars = counts.plot(kind='bar', legend=False, ax=ax)
    for bar, color in zip(bars.patches, value_colours):
        bar.set_color(color)
    fig.patch.set_facecolor('none')
    ax.tick_params(labelcolor='white')
    ax.set_facecolor('none') 
    ax.set_xlabel("Classification")
    ax.set_ylabel("Number of Classifications")
    st.pyplot(fig)  



  

retreive = "SELECT * FROM Sheet1"
select=conn.query(sql=retreive,ttl=20)
df=pd.DataFrame(select)
fake = df[df['Classification'] == 'Fake']['Article']
real = df[df['Classification'] == 'Real']['Article']

st.header('Most common words in Fake News articles:')
text = " ".join(t for t in fake)
word_cloud = WordCloud(collocations = False, background_color = 'black').generate(text)
fig, ax = plt.subplots()
ax.imshow(word_cloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)  

st.header('Most common words in Real News articles:')
text = " ".join(t for t in real)
word_cloud = WordCloud(collocations = False, background_color = 'black').generate(text)
fig, ax = plt.subplots()
ax.imshow(word_cloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)   

#all
st.header('Sentiment Analysis:')
option = st.selectbox(
    "Find out the Sentiments of the Articles that our model has classified",
    ("All Articles", "Fake Articles", "Real Articles"))

if option == 'All Articles':
    retreive = "SELECT Sentiment FROM Sheet1"
    select=conn.query(sql=retreive,ttl=20)
    counts=select.value_counts().sort_values(ascending=True) #sorts the counts of each sentiment
    fig, ax = plt.subplots(figsize=(11, 6))
    colours = {('Negative',):'#ff4c4c',('Positive',):'#38d864',('Neutral',):'#ffff00'}
    value_colours = [colours[x] for x in counts.index] #makes sure that colours stay the same for sentiments    
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')   
    counts.plot.pie(ax=ax,autopct='%1.1f%%', colors=value_colours,textprops={'color':"black"}, labels=None)
    ax.axis('equal')
    st.pyplot(fig)

elif option == 'Fake Articles':
    retreive = "SELECT Sentiment FROM Sheet1 WHERE Classification == 'Fake'"
    select=conn.query(sql=retreive,ttl=20)
    counts=select.value_counts().sort_values(ascending=True) #sorts the counts of each sentiment
    fig, ax = plt.subplots(figsize=(11, 6))
    colours = {('Negative',):'#ff4c4c',('Positive',):'#38d864',('Neutral',):'#ffff00'}
    value_colours = [colours[x] for x in counts.index] #makes sure that colours stay the same for sentiments    
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')   
    counts.plot.pie(ax=ax,autopct='%1.1f%%', colors=value_colours,textprops={'color':"black"}, labels=None)
    ax.axis('equal')
    st.pyplot(fig)

elif option == 'Real Articles':
    retreive = "SELECT Sentiment FROM Sheet1 WHERE Classification == 'Real'"
    select=conn.query(sql=retreive,ttl=20)
    counts=select.value_counts().sort_values(ascending=True) #sorts the counts of each sentiment
    fig, ax = plt.subplots(figsize=(11, 6))
    colours = {('Negative',):'#ff4c4c',('Positive',):'#38d864',('Neutral',):'#ffff00'}
    value_colours = [colours[x] for x in counts.index] #makes sure that colours stay the same for sentiments    
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')   
    counts.plot.pie(ax=ax,autopct='%1.1f%%', colors=value_colours,textprops={'color':"black"}, labels=None)
    ax.axis('equal')
    st.pyplot(fig)