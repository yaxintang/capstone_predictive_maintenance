import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import duckdb
from sklearn.model_selection import train_test_split
import streamlit as st
import st_function as f

#repeat for every page to alter the default
st.set_page_config(
    page_title = "Predictive Maintenance - EDA",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()

with duckdb.connect("data/team_data.duckdb") as conn:
    df = conn.execute("SELECT * FROM df_train").fetchdf()
    
#df.head()  

st.session_state.df = df


lab_dict = {
    "udi":"UDI",
    "product_id":"Product ID",
    "type":"Type",
    "air_temperature_k":"Air temperature [K]",
    'process_temperature_k':'Process temperature [K]',
    'rotational_speed_rpm':'Rotational speed [RPM]',
    'torque_nm':'Torque [Nm]',
    'tool_wear_min':'Tool wear [min]',
    "machine_failure":"Machine failure",
    "twf":"TWF",
    "hdf":"HDF",
    "pwf":"PWF",
    "osf":"OSF",
    "rnf":"RNF"
    }


#pio.templates.default = "seaborn"

pio.templates["custom_theme"] = go.layout.Template(
    layout=go.Layout(
        #paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor="rgba(230,230,230,255)",

        colorway=px.colors.qualitative.D3

    )
)

pio.templates.default = 'custom_theme'

st.title("📊 EDA Visualization")
st.write("")
st.subheader("Height vs Weight")