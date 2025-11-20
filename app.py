import streamlit as st
import pandas as pd
import st_function as f

# Set configuration and sidebar navigation
st.set_page_config(
    page_title = "Predictive Maintenance",
    page_icon=":gear:",
)

f.navigation()

# Set Title
st.title("Predictive Maintenance")

# Insert image
st.image('images/milling.JPG', width=500)

st.subheader("What are we going to do?")
st.write("")
st.markdown('- EDA Visualizations \n - Prediction: Is the machine need maintenance?')

# Save variables in the session state
st.session_state.df = pd.read_csv("data/ai4i2020.csv")