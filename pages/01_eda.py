import streamlit as st
import pandas as pd
import numpy as np
import st_function as f

f.navigation()

# Set Title
st.title("EDA")

st.dataframe(st.session_state.df)