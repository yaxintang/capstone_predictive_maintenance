import pandas as pd
import duckdb
import streamlit_pages.st_function as f
import plotly.express as px
import streamlit as st
import pickle
import math
import shap
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

model_name="random_forrest_optimized"
model = pickle.load(open(f"models/{model_name}.sav", 'rb'))

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

def display_value():
    with col1:
        air_temp = st.session_state.ss_process_temperature_k - st.session_state.ss_delta_temp_k
        #st_air_temperature_k = st.slider(label='Air temperature [K]', min_value=295, max_value=304, value=air_temp, step=1 , key='ss_air_temperature_k', disabled=True)
        st.write("Air temperature: {air_temp}")

################## CONTENT START

############ TITLE

# Set Title
st.title(":bulb: Digital Twin")

st.markdown("In this digital twin environment different machine settings can be tested before application. Machine failure will be predicted based on input settings using a model based on historic data. Additionally, the feature with highest importance to the prediction result will be reported." )

st.subheader("Input Machine Settings")
col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="top", border=False, width="stretch")


with col1:

    st_process_temperature_k  = st.slider(label='Process temperature [K]', min_value=305, max_value=313, value=309 , step=1 , key='ss_process_temperature_k')#, on_change=display_value)
    #st_delta_temp_k  = st.slider(label='Temperature Difference [K]', min_value=7, max_value=12, value=9 , step=1 , key='ss_delta_temp_k')#, on_change=display_value)
    st_air_temperature_k  = st.slider(label='Air temperature [K]', min_value=295, max_value=304, value=299, step=1 , key='st_air_temperature_k')#, disabled=True)

with col2:
    st_rotational_speed_rpm  = st.slider(label='Rotational speed [RPM]', min_value=1168, max_value=2886, value=2027 , step=1 , key='st_rotational_speed_rpm')
    st_torque_nm  = st.slider(label='Torque [Nm]', min_value=3, max_value=76, value=40 , step=1 , key='st_torque_nm')
    #st_power_w  = st.slider(label='Power [W]', min_value=1148, max_value=10469, value=5809 , step=1 , key='st_power_w')

with col3:
    st_type = st.select_slider(label='Product quality', options=["L","M","H"], value="M", key='st_type')

    st_tool_wear_min  = st.slider(label='Tool usage [min]', min_value=0, max_value=253, value=126 , step=1 , key='st_tool_wear_min')

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

#X_test_dict

st.subheader("Predicted Result")

X_test = pd.DataFrame.from_dict(X_test_dict)

#X_test
y_pred = model.predict(X_test)



# just here for debug
#st.write(X_test.transpose())

if y_pred == 0:
    st.markdown("👍 Under the provided parameters machine is predicted to run in without failure.")
else:
    st.markdown("👎 Under the provided parameters machine is predicted to fail.")

st.subheader("Influential Parameters")

#shap values
# 1) fitted pipeline from RandomizedSearchCV
fitted_pipe = model
# 2) split pipeline
#from sklearn.utils.validation import check_is_fitted
preprocess = fitted_pipe.named_steps["preprocess"]
rf_model   = fitted_pipe.named_steps["random_forest"]

# ensure RandomForest is fitted
#check_is_fitted(rf_model, "estimators_")
# 3) transform X_test
Xte_trans = preprocess.transform(X_test)

# feature names
try:
    feat_names = preprocess.get_feature_names_out() # some version of SKlearn will not support get_feature_names_out, so we do this step
except:
    feat_names = [f"f_{i}" for i in range(Xte_trans.shape[1])] # if get_feature_names_out is not supported then features name will be f_1

# convert to dense DataFrame
if hasattr(Xte_trans, "toarray"):
    Xte_trans = Xte_trans.toarray()
Xte_trans = pd.DataFrame(Xte_trans, columns=feat_names, index=X_test.index)

# 4) SHAP values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(Xte_trans)

# take class 1 SHAP values: shape -> (n_samples, n_features)
sv_class1 = shap_values.values[:, :, 1]
#st.write(sv_class1.transpose())
# 5) SHAP table
mean_abs = np.abs(sv_class1).mean(axis=0)
shap_table_transformed = (
    pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
      .sort_values("mean_abs_shap", ascending=False)
      .reset_index(drop=True)
)
shap_dict = {
"num__power" : "Power",
"num__rotational_speed_rpm"  :"Speed",
"num__delta_temperature" : "Temperature - delta",
"num__torque_nm" : "Torque",
"num__tool_wear_min" : "Tool wear",
"cat__type_L": "Product type - L",
"num__process_temperature_k": "Temperature - process",
"cat__type_M": "Product type - L",
"num__air_temperature_k": "Temperature - air",
"cat__type_H": "Product type - H",
}
top_3 = shap_table_transformed.head(3).transpose()
chart_width = 600
st.plotly_chart(f.chart_shap(top_3, shap_dict),width=chart_width)
