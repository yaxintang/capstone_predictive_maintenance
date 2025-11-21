import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
#import duckdb
from sklearn.model_selection import train_test_split
#import streamlit as st

def import_stuff():
    import plotly.express as px


def navigation():
    import streamlit as st
    """
    Function to customize navigation sidebar panel
    """
    
    st.sidebar.page_link("app.py", label='👋 Welcome')
    st.sidebar.page_link("pages/proj_management.py", label="📅 Project Management")
    st.sidebar.page_link("pages/01_eda.py", label="📊 EDA")
    #st.sidebar.page_link("pages/02_pipeline.py", label="📜 Pipeline")

