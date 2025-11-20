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
    st.session_state.df_train = conn.execute("SELECT * FROM df_train").fetchdf()
    
#df.head()  

#st.session_state.df_train = df_train
df_train = st.session_state.df_train
df_train["machine_failure"] = df_train["machine_failure"].astype("category")

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

################## CONTENT START

############ TITLE

st.title("📊 Exploratory Data Analysis")



###### SUBHEADER

st.write("")
st.subheader("Training Dataset")
st.markdown("""
    - **split:** training dataset represents  70% of all observations
    - **shape:** 14 features x 7.000 observations
    - **categorical values:** xyz
    - **numerical values:** abc
    
    """)
df_train
###### SUBHEADER

st.write("")
st.subheader("Histogram of Numerical Data")
st.markdown("Text...")

#col_sort = "udi"
col_grouping = ["machine_failure"]
plot_list = [
    "air_temperature_k", 
    "process_temperature_k", 
    "rotational_speed_rpm", 
    "torque_nm", 
    "tool_wear_min",
    "machine_failure"
]
size = 500
df_chart = df_train#.sort_values(col_sort)

for column in plot_list:
    for group in col_grouping:
        fig = px.histogram(
            data_frame = df_chart,
            x = column,
            color = group,
            width = size,
            height = size,
            opacity=0.7,
            barmode="overlay",
            labels=lab_dict,
            title=f"Histogram of {lab_dict[column]}"
            )
        fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.20,
        ))
        #fig.show()
        st.plotly_chart(fig)


###### SUBHEADER

st.write("")
st.subheader("Box Plot of Selected Data")
st.markdown("Observation outside inter quartile range (IQR) will not be removed as the are part of the valid data distribution.")

size = 400

cols = [ 'air_temperature_k',
 'process_temperature_k',
 'rotational_speed_rpm',
 'torque_nm',
 'tool_wear_min',]
df_chart = df_train
for col in cols:
    fig = px.box(
        df_chart, 
        y=col, 
        width=size, 
        height=size, 
        labels=lab_dict,
        )
    #fig.show()
    st.plotly_chart(fig)


###### SUBHEADER

st.write("")
st.subheader("Scatter Plot of Selected Data")


# individual scatter plots
size = 400
scatter_color ="machine_failure"
df_chart = df_train#[df_train["machine_failure"]==1]

scatter_pairs = {
    1:[ 'rotational_speed_rpm','torque_nm',  scatter_color],
    #2:[ 'torque_nm', 'tool_wear_min', scatter_color],
    3:[ 'torque_nm', 'tool_wear_min',  scatter_color],
    #4:['cons_number', 'udi',  scatter_color],
    #5:[ 'udi','torque_nm',  scatter_color],
    #6:[ 'cons_number','torque_nm',  scatter_color]
    7:["air_temperature_k", "process_temperature_k",scatter_color ]
    }


for key, values in scatter_pairs.items():
    fig = px.scatter(
        df_chart,
        x = values[0],
        y = values[1],
        color = values[2],
        opacity = 0.7,
        width=size,
        height=size,
        labels=lab_dict,
        title=f"{lab_dict[values[0]]} vs {lab_dict[values[1]]}"

        #        color_continuous_scale = ["green", "red"]
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.30,
   ))

    #fig.show()
    st.plotly_chart(fig)


