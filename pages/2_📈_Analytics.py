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

st.header('Analytics📈')

conn = st.connection("gsheets",type=GSheetsConnection,show_spinner=False)
sql="""
SELECT COUNT(Classification) FROM Sheet1 where Classification == 'Fake'
""" #uses sql select statement to get updated result
select=conn.query(sql=sql,ttl=20)
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
    "See how many fake and real articles our detector has classified over different time periods",
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
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

elif option == 'Last 14 Days':
    fourtheenDays = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 14 
    """
    select=conn.query(sql=fourtheenDays,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

elif option == 'Last Month':
    oneMonth = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 30 
    """
    select=conn.query(sql=oneMonth,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

elif option == 'Last 6 Months':
    sixMonths = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 180 
    """
    select=conn.query(sql=sixMonths,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

elif option == 'Last Year':
    oneYear = """
    SELECT *
    FROM Sheet1
    WHERE CAST(Date AS DATE) >= CURRENT_DATE - 356 
    """
    select=conn.query(sql=oneYear,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

elif option == 'All Time':
    total = 'SELECT* FROM Sheet1'
    select=conn.query(sql=total,ttl=20)
    counts=pd.DataFrame(select['Classification'].value_counts())
    chart_data = pd.DataFrame(counts)
    st.bar_chart(data=chart_data, color='#0f5bd1')

