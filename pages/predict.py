import pandas as pd
import st_function as f
import plotly.express as px
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio



# Set configuration and sidebar navigation
st.set_page_config(
    page_title = "Predictive Maintenance - Digital Twin",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()

# plotly chart zemplate
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

# load fitted model
model_name="random_forrest_optimized"
model = pickle.load(open(f"models/{model_name}.sav", 'rb'))

# define names in charts
txt = {"reduce":'reducing probability of failing',"increase": 'increasing probability of failing', "sum": "sum of SHAPs"}
shap_dict = {
"cat__type_H": "Product type (H)",
"cat__type_L": "Product type (L)",
"cat__type_M": "Product type (M)",
"num__air_temperature_k": "Temperature (air)",
"num__process_temperature_k": "Temperature (process)",
"num__delta_temperature" : "Temperature (delta)",
"num__rotational_speed_rpm"  :"Speed",
"num__torque_nm" : "Torque",
"num__tool_wear_min" : "Tool wear",
"num__power" : "Power",
"sum" : txt["sum"]
}


################## CONTENT START

############ TITLE

# Set Title
st.title(":bulb: Digital Twin")

st.markdown("In this digital twin environment different machine settings can be tested before application. Machine failure will be predicted based on input settings using a model based on historic data. Additionally, the feature with highest importance to the prediction result will be reported." )


###### SUB

st.subheader("General Feature Importance of the Model")

cola, colb = st.columns(2, gap="small", vertical_alignment="top", border=True, width="stretch")

    ###### COL A

with cola:
    # chart general feature importance
    st.plotly_chart(f.chart_feature_importance(model, shap_dict) )

    ###### COL B

with colb:
    st.markdown("- Features of the model have a different impact for classification. " )
    st.markdown("- General feature importance for random forest is based on mean decrease in impurity (MDI)." )
    st.markdown('- Power (Torque x Rotational Speed) is the most important feature of this model.' )
    st.markdown('- Product types are the least important features of this model.' )

###### SUB

st.subheader("Feature Importance for Prediction")
st.markdown("For a specific prediction the importance of a feature depends on it value and the interaction with other features. Below shapley additive explanations (SHAP) for the prediction of selected machine settings.")

    ###### COL 1

col1, col2 = st.columns(2, gap="small", vertical_alignment="top", border=True, width="stretch")
with col1:

        ###### SUB

    st.subheader("Input Machine Settings")

    # sliders
    st_process_temperature_k  = st.slider(label='Process temperature [K]', min_value=305, max_value=313, value=309 , step=1 , key='ss_process_temperature_k')#, on_change=display_value)
    st_air_temperature_k  = st.slider(label='Air temperature [K]', min_value=295, max_value=304, value=299, step=1 , key='st_air_temperature_k')#, disabled=True)
    st_type = st.select_slider(label='Product quality', options=["L","M","H"], value="M", key='st_type')
    st_rotational_speed_rpm  = st.slider(label='Rotational speed [RPM]', min_value=1168, max_value=2886, value=2027 , step=1 , key='st_rotational_speed_rpm')
    st_torque_nm  = st.slider(label='Torque [Nm]', min_value=3, max_value=76, value=40 , step=1 , key='st_torque_nm')
    st_tool_wear_min  = st.slider(label='Tool usage [min]', min_value=0, max_value=253, value=126 , step=1 , key='st_tool_wear_min')
    
    # define X_test based on input
    X_test_dict = {
        "type":[st_type],
        "air_temperature_k" : [int(st_air_temperature_k)],
        "process_temperature_k" : [int(st_process_temperature_k)],
        "delta_temperature" : [int(st_process_temperature_k)-int(st_air_temperature_k)], 
            "rotational_speed_rpm" : [int(st_rotational_speed_rpm)],
            "torque_nm" : [int(st_torque_nm)],
            "tool_wear_min" : [int(st_tool_wear_min)],
            "power" : [int(st_torque_nm) * int(st_rotational_speed_rpm)]
            }
    X_test = pd.DataFrame.from_dict(X_test_dict)
    
    
    

    ###### COL 2
    
with col2:
    
        ###### SUB
    
    st.subheader("Predicted Results")
    
    # predict based on input
    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[0]
    prob = int(pred_proba[y_pred]*100)
    if y_pred == 0:
        text = "✅ Safe Operations"
    else:
        text = "⚠️ Machine Failure"
    
    st.write(f"With probability of **{prob}%** these settings will result in:")
    st.markdown(f'<div style="font-weight: bold;text-align: center">{text}</div>',  unsafe_allow_html=True)
    
    # Shapley Additive Explanations (SHAP)
    st.plotly_chart(f.chart_shap(model, 3, txt,shap_dict, X_test ))