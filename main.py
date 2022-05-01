import Predict
import Dataset
import Intro
import streamlit as st
PAGES={
    "Introduction": Intro,
    "Dataset": Dataset,
    "Predict":Predict,
    
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()