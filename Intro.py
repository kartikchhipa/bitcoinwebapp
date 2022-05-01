# app2.py
import streamlit as st
from PIL import Image
def app():
    image=Image.open('Crypto-News-February-23-Bitcoin-will-continue-to-decline-in.jpeg')
    st.title('Introduction')
    st.image(image)
    st.write('This project is developed as a part of course project for the Pattern Recognition and Machine Learning under Prof. Richa Singh. This Project\
            is all about Bitcoin Price Prediction using Bitcoin Historical Price Data and applying various Machine Learning Principle and models\
            to accurately predict the future price of Bitcoin to some extent. However in the present time Bitcoin is one of the most volatile cryptocurrency\
            and the prices fluctuate very rapidly. So even the most robust models may fail to predict the price of Bitcoin.')
    