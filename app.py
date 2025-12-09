import pandas as pd
import duckdb
import st_function as f
import plotly.express as px
import streamlit as st


# Set configuration and sidebar navigation
st.set_page_config(
    page_title = "Predictive Maintenance - Welcome",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()


# Save variables in the session state
#st.session_state.df = pd.read_csv("data/ai4i2020.csv")
with duckdb.connect("data/team_data.duckdb") as conn:
    st.session_state.df_train = conn.execute("SELECT * FROM df_train").fetchdf()
df_train = st.session_state.df_train

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


################## CONTENT START

############ TITLE

# Set Title
st.title("Predictive Maintenance")

# Insert image
st.image('images/milling.jpg', width=500)

###### SUBHEADER

st.write("")
st.subheader("Project Definition")
st.markdown('In many industrial environments, assets are maintained using a combination of reactive (“fix it when it breaks”) '
'and preventive (time-based) methods. This hybrid approach still leads to **unplanned downtime (reactive)**, **excessive spare-part consumption (preventive)**, and **higher labor costs**. ')
st.markdown('As industries move toward **Industry 4.0**, traditional reactive and preventive maintenance approaches are no longer sufficient to maintain high-performance manufacturing systems.' \
'The objective of this project is to build a predictive maintenance model that uses sensor data and analytics to identify failure patterns before they occur.')

###### SUBHEADER

st.write("")
st.subheader("Baseline Model")
st.markdown('Baseline model will be represented by a simple decision tree outlining safe operation modes based on two main features (see below).')
st.markdown('**Decision Tree goes here**')
st.write("")

st.markdown('F1 score of baseline model is: XYZ')
st.markdown('**Confusion matrix goes here**')

###### SUBHEADER

st.write("")
st.subheader("Value Proposition")
#st.markdown('What problem does the product solve?')
st.markdown('A predictive maintenance system will \n'
'- reduces downtime \n'
'- lowers maintenance costs \n '
'- extends equipment lifetime \n'
'- and enables data-driven maintenance decisions \n'
' \n delivered through a scalable, easily deployable ML pipeline.')

###### SUBHEADER

st.write("")
st.subheader("Model Metrics")
#st.markdown('- Option 1:')
#st.markdown('Recall x Availability: Building on preventive maintenance, failure events are rare and expensive, a model should predict as many failures as possible correctly.')
#st.markdown('Recall = True Positives / (True Positives + False Negatives)')
#st.markdown('- Option 2:')
st.markdown('The target of the model is to correctly classify whether there will be a failure, hence it is the target (failure = 1).')
st.write(' ')
st.markdown(
    'We want the model to: \n'
    '1. maximize identification of failures by minimizing "false negatives" (recall) \n'
    '1. maximize tool run-time by minimizing "false positives" (precision)')
st.markdown('We will therefore go with the F1 score as it provides a balance between recall and precision.')
st.write(' ')
st.write(' ')
st.markdown('$$ Recall = { True Positives \over  True Positives + False Negatives} $$')
st.markdown('$$ Precision = { True Positives \over  True Positives + False Positives} $$')
st.markdown('$$ F1 Score = 2 * {Precision * Recall \over Precision + Recall} $$')

###### SUBHEADER
st.write("")
st.subheader("What are we going to do?")
st.markdown('- EDA Visualizations \n - Pipeline \n - Prediction: Is the machine need maintenance?')

