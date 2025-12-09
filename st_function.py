import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
#import duckdb
from sklearn.model_selection import train_test_split
import streamlit as st
#from typing import Union
#import base64 # Neu: Für die Kodierung des HTML-Inhalts


def navigation():
    """
    Function to customize navigation sidebar panel
    """
    
    #st.sidebar.page_link("st_app.py", label='👋 Welcome')
    #st.sidebar.page_link("streamlit_pages/proj_management.py", label="📅 Project Management")
    #st.sidebar.page_link("eda.py", label="📊 EDA")
    #st.sidebar.page_link("predict.py", label="💡 Digital Twin")

def chart_hist_box(data_frame, column, group, lab_dict):
    fig = px.histogram(
        data_frame = data_frame,
        x = column,
        color = group,
        opacity=0.7,
        barmode="overlay",
        labels=lab_dict,
        title=f"Histogram of {lab_dict[column]}",
        marginal="box",
        
        )
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="top",
    y=-0.20,
    ))
    return fig

def chart_scatter(data_frame, x, y, color, lab_dict):
    fig = px.scatter(
        data_frame,
        x = x,
        y = y,
        color = color,
        opacity = 0.7,
        labels=lab_dict,
        title=f"{lab_dict[x]} vs {lab_dict[y]}"
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.30,
    ))
    return fig

def chart_corr(data_frame,features_included,lab_dict ):
    corr=data_frame[features_included].corr()
    fig = px.imshow(
        corr, 
        text_auto=".1f", 
        color_continuous_scale=[px.colors.qualitative.D3[0],"white",px.colors.qualitative.D3[1]], 
        zmin=-1,  
        zmax=1,
        x=[lab_dict[x] for x in features_included],
        y=[lab_dict[x] for x in features_included]

    )
    return fig

def chart_shap(data_frame,lab_dict ):
    fig = px.bar(
        data_frame,
        labels=lab_dict
    )
    return fig


## function to wrap an iFrame around plotly chart for a streamlit reveal slide
#def embed_plotly_in_iframe(fig: Union[go.Figure, dict], height: int = 550) -> str:
#   
#    chart_html_full = fig.to_html(
#        include_plotlyjs='cdn', 
#        full_html=True,
#        div_id='plotly_reveal_embed'
#    )
#    
#   
#    html_bytes = chart_html_full.encode('utf-8')
#    encoded = base64.b64encode(html_bytes).decode('utf-8')
#    data_uri = f"data:text/html;base64,{encoded}"
#    
#   
#    iframe_tag = f'<iframe src="{data_uri}" style="width:100%; height:{height}px; border:none; background-color: grey;"></iframe>'
#    
#    return iframe_tag

## create in a streamlit reveal slide with predefined content
#def slide_chart_comment(df_chart, col_grouping, plot_list,lab_dict ):
#    markdown_content =  ""
#    for key, value in plot_list.items():
#        markdown_content = markdown_content+f"""
#        ## {lab_dict[value[0]]}
#        {embed_plotly_in_iframe(chart_hist_box(df_chart, value[0], col_grouping, lab_dict))}
#        {value[1]}
#        ---
#        """
#    markdown_content = markdown_content+f"""
#    ## End
#    """
#    return markdown_content