import pandas as pd
import numpy as np
import plotly.express as px
import shap
import streamlit as st

def navigation():
    """
    Function to customize streamlit navigation sidebar panel
    """
    
    st.sidebar.page_link("st_app.py", label='👋 Welcome')
    st.sidebar.page_link("pages/proj_management.py", label="📅 Project Management")
    st.sidebar.page_link("pages/eda.py", label="📊 EDA")
    st.sidebar.page_link("pages/predict.py", label="💡 Digital Twin")

def chart_hist_box(data_frame, column, group, lab_dict):
    """
    Generate a histogram chart using plotly

    Args:
        data_frame (df): data frame
        column (str): column to plot
        group (str): chart grouping
        lab_dict (dict): nice column names

    Returns:
        chart: plotly chart
    """
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
    """   
    Generate a scatter plot using plotly

    Args:
        data_frame (df): data frame
        x (str): column to plot
        y (str): column to plot
        color (str): chart grouping
        lab_dict (dict): nice column names

    Returns:
        chart: plotly chart
    """
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
    """
    Generate a correlation chart using plotly

    Args:
        data_frame (df): data frame
        features_included (list): list of column names
        lab_dict (dict): nice column names

    Returns:
        chart: plotly chart
    """
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

def chart_shap(model,top ,txt,shap_dict, X_test):
    """
    Generate a shap chart using plotly

    Args:
        model (model): fitted model
        top (int): number of top items
        txt (dict): chart text
        shap_dict (dict): nice column names
        X_test (df): data frame for prediction

    Returns:
        chart: plotly chart
    """

    # get model steps and preprocess data
    preprocess = model.named_steps["preprocess"]
    rf_model   = model.named_steps["random_forest"]
    Xte_trans = preprocess.transform(X_test)
    try:
        feat_names = preprocess.get_feature_names_out() # some version of SKlearn will not support get_feature_names_out, so we do this step
    except:
        feat_names = [f"f_{i}" for i in range(Xte_trans.shape[1])] # if get_feature_names_out is not supported then features name will be f_1
    if hasattr(Xte_trans, "toarray"):
        Xte_trans = Xte_trans.toarray()
    Xte_trans = pd.DataFrame(Xte_trans, columns=feat_names, index=X_test.index)

    # get shap values for class 1
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(Xte_trans)
    sv_class1 = shap_values.values[0, :, 1]
    data_frame = pd.DataFrame({"feature": feat_names, "shap": sv_class1}).sort_values("shap", ascending=False)

    # add additional columns for charting
    data_frame['Group'] = np.where(data_frame['shap'] < 0, txt["reduce"] ,txt["increase"]  )
    data_frame["shap_abs"] = np.abs(data_frame["shap"])
    data_frame['show'] = np.where(data_frame["shap_abs"] > data_frame.sort_values("shap_abs", ascending=False)["shap_abs"].iloc[top], 1, 0 )
    
    row_sum=len(data_frame)
    data_frame.loc[row_sum]=data_frame.sum()
    data_frame["feature"].loc[row_sum]="sum"
    data_frame["Group"].loc[row_sum]=txt["sum"]
    data_frame["show"].loc[row_sum]=1
    
    data_frame["long_name"]=data_frame["feature"].replace(shap_dict)
    data_frame = data_frame[data_frame["show"]==1][["long_name", "shap", "show", "Group"]]
    
    fig = px.bar(
        data_frame,
        color_discrete_map={txt["increase"]: 'red',  txt["reduce"]: 'green',txt["sum"]: "blue"},
        color='Group',
        orientation='h',
        x="shap",
        y="long_name",
    )
    fig.update_layout(
        legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.30,             
    ),
    #xaxis_tickfont_size= 25,
    #xaxis_title_font_size =25,
    #yaxis_tickfont_size= 25,
    #yaxis_title_font_size =25,
    #legend_font_size = 25,

    #height =500,

    xaxis=dict(range=[-0.75, 0.75]),
     #   showlegend=False,
        xaxis_title=f"Top {top} Influential Features",
        yaxis_title="SHAP Value"
    )
    return fig

def chart_feature_importance(model, dict_labels):
    """
    Generate a shap chart using plotly

    Args:
        model (model): fitted model
        dict_labels (dict): nice column names

    Returns:
        chart: plotly chart
    """
    rf_model   = model.named_steps["random_forest"]
    feats = {}
    for feature, importance in zip(dict_labels, rf_model.feature_importances_*100):
        feats[feature] = importance
    
    df = pd.DataFrame(list(feats.items()), columns=['feature', 'value'])
    df["label"]=df["feature"].replace(dict_labels)
    
    fig=px.bar(df
               ,orientation="h",
               y="label",
               x="value")
    fig.update_layout(
        xaxis_title="Importance [%]",
        yaxis_title="Feature" ,
        showlegend = False,

        #xaxis_tickfont_size= 25,
        #xaxis_title_font_size =25,
        #yaxis_tickfont_size= 25,
        #yaxis_title_font_size =25

        )
    return fig
