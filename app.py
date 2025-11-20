import streamlit as st
import pandas as pd

st.title("Predictive Maintenance")

st.image('images/milling.JPG')

st.subheader("What are we going to do?")
st.write("")
st.markdown('- EDA Visualizations \n - Prediction: Is the machine need maintenance?')

st.set_page_config(
    page_title = "Predictive Maintenance",
    page_icon=":gear:",
    layout="wide"
)

st.session_state.df = pd.read_csv("data/ai4i2020.csv")
st.dataframe(st.session_state.df)