import pickle
import streamlit as st
import pandas as pd
import numpy as np
def app():
    data=pd.read_csv('https://raw.githubusercontent.com/KartikChhipa01/datasets/main/BTC-USD.csv')
    data['Date']=pd.to_datetime(data['Date'])
    
    st.dataframe(data)