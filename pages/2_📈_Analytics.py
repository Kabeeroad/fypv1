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

conn = st.connection("gsheets",type=GSheetsConnection)
sql='SELECT* FROM Sheet1' #uses sql select statement to get updated result
select=conn.query(sql=sql,ttl=20)
st.dataframe(select)