import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import duckdb
from sklearn.model_selection import train_test_split
import streamlit as st
import st_function as f
import math
from pandas import set_option
import reveal_slides as rs


#repeat for every page to alter the default
st.set_page_config(
    page_title = "Predictive Maintenance - EDA",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()

# get data
with duckdb.connect("data/team_data.duckdb") as conn:
    st.session_state.df_train = conn.execute("SELECT * FROM df_train").fetchdf()
    
#st.session_state.df_train = df_train
df_train = st.session_state.df_train

# generate additional columns
# power measured in W = 1Nm/s // torque in Nm // rotational speed in rpm => * 60 (sec) / 2pi (one ration)
df_train["power_w"] = (df_train["torque_nm"] / (df_train["rotational_speed_rpm"] * 60 / (2 * math.pi))) * 1000
df_train["delta_temp_k"] = df_train["process_temperature_k"]-df_train["air_temperature_k"]

# set categorical columns
cat_cols =[
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf"]

for col in cat_cols:
    df_train[col] = df_train[col].astype("category")

# define dictionary for labels
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
    "twf":"Tool Wear Failure",
    "hdf":"Heat Dissipation  Failure",
    "pwf":"Power Failure",
    "osf":"Over Strain Failure",
    "rnf":"Random Failure",
    "delta_temp_k": "Temperature Difference [K]",
    "power_w":"Power [kW]",
    "cons_number":"#Product"
    }



#pio.templates.default = "seaborn"

pio.templates["custom_theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(14,17,23,255)', 
        plot_bgcolor='rgba(14,17,23,255)',
        font_color="white",
        font_size = 18,
        colorway=px.colors.qualitative.D3

    )
)

pio.templates.default = 'custom_theme'

chart_width = 600


################## CONTENT START

############ TITLE

st.title("📊 Exploratory Data Analysis")

###### SUBHEADER

st.write("")
st.subheader("Training Dataset")
st.markdown("""
    - **split:** training dataset represents  70% of all observations
    - **shape:** 8 features x 7.000 observations
    - **categorical values:** Type and all failure categories
    - **numerical values:**  Air temperature [K], Process temperature [K], Rotational speed [RPM], Torque [Nm], Tool wear [min]    
    """)
st.dataframe(df_train.rename(columns=lab_dict), width=chart_width)

###### SUBHEADER

st.write("")
st.subheader("Distribution of Numerical Data")
st.markdown("Observations outside inter quartile range (IQR) will not be removed as they are part of the valid data distribution.")


# generate plots and text 

# 1. feature to plot 2. comment on plot
plot_list = {
    #"air_temperature_k", 
    1:["process_temperature_k", "Higher process temperature increases probability of machine failure."],
    2:["rotational_speed_rpm", "Low rotational speed increases probability of machine failure."],
    3:["torque_nm", "Higher torque increases probability of machine failure."],
    4:["power_w", "Higher machine power (torque / rotation speed ) increases probability of machine failure."],
    5:["machine_failure", "Dataset is imbalanced (~30:1)"]    
}

# grouping by
col_grouping = "machine_failure"

df_chart = df_train#.sort_values(col_sort)

# charts in streamlit
for key, value in plot_list.items():
    st.plotly_chart(f.chart_hist_box(df_chart, value[0], col_grouping, lab_dict), width=chart_width)
    
    st.markdown(f"**Note:** {value[1]}")

# reveal slides
markdown_content = f.slide_chart_comment(df_chart, col_grouping, plot_list,lab_dict)
rs.slides(
    markdown_content, 
    theme="black" 
)
###### SUBHEADER

st.write("")
st.subheader("Scatter Plot of Selected Data")


# generate plots and text 
df_chart = df_train#[df_train["machine_failure"]==1]

# list in dict: 1. x value 2. y value 3. grouping (color) 4. comment
scatter_pairs = {    
    1:[ 'rotational_speed_rpm','torque_nm',  "pwf", "Higher machine power (torque / rotation speed) increases probability of failure."],
    2:["tool_wear_min", 'torque_nm', "osf", "High torque combined with long tool wear time increases probability of failure." ]
    }

for key, value in scatter_pairs.items():
    st.plotly_chart(f.chart_scatter(df_chart, value[0], value[1],value[2], lab_dict), width= chart_width)
    
    st.markdown(f"**Note:** {value[3]}")
    st.markdown("")


###### SUBHEADER

st.write("")
st.subheader("Correlation matrix")
st.markdown("- **Highly negatively correlated columns:** \n" \
            "   - torque vs rotational speed\n"
            "- **Highly positively correlated columns:** \n" \
            "   - air temperature vs process temperature"
            )
features_included = ["air_temperature_k","process_temperature_k","rotational_speed_rpm","torque_nm","tool_wear_min"]

df_chart = df_train#[df_train["machine_failure"]==1]
st.plotly_chart(f.chart_corr(df_chart, features_included,lab_dict),width=chart_width)

