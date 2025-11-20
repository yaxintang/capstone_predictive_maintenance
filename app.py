import streamlit as st
import pandas as pd
import st_function as f
import duckdb

# Set configuration and sidebar navigation
st.set_page_config(
    page_title = "Predictive Maintenance - Welcome",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()

################## CONTENT START

############ TITLE

# Set Title
st.title("Predictive Maintenance")

# Insert image
st.image('images/milling.jpg', width=500)

###### SUBHEADER

st.subheader("Project Definition")
st.markdown('In many industrial environments, assets are maintained using a combination of reactive (“fix it when it breaks”) '
'and preventive (time-based) methods. This hybrid approach still leads to **unplanned downtime (reactive)**, **excessive spare-part consumption (preventive)**, and **higher labor costs**. ')
st.markdown('As industries move toward **Industry 4.0**, traditional reactive and preventive maintenance approaches are no longer sufficient to maintain high-performance manufacturing systems.' \
'The objective of this project is to build a predictive maintenance model that uses sensor data and analytics to identify failure patterns before they occur.')

###### SUBHEADER

st.subheader("Value Proposition")
#st.markdown('What problem does the product solve?')
st.markdown('A predictive maintenance system will \n'
'- reduces downtime \n'
'- lowers maintenance costs \n '
'- extends equipment lifetime \n'
'- and enables data-driven maintenance decisions \n'
' \n delivered through a scalable, easily deployable ML pipeline.')

###### SUBHEADER

st.subheader("Model Metrics")
st.markdown('- Option 1:')
st.markdown('Recall x Availability: Building on preventive maintenance, failure events are rare and expensive, a model should predict as many failures as possible correctly.')
st.markdown('Recall = True Positives / (True Positives + False Negatives)')
st.markdown('- Option 2:')
st.markdown('F1 Score forces the model to care about both recall and precision.')
st.markdown('F1 Score= 2 * (Precision * Recall) / (Precision + Recall)')

###### SUBHEADER

st.subheader("What are we going to do?")
st.write("")
st.markdown('- EDA Visualizations \n - Pipeline \n - Prediction: Is the machine need maintenance?')

# Save variables in the session state
#st.session_state.df = pd.read_csv("data/ai4i2020.csv")
with duckdb.connect("data/team_data.duckdb") as conn:
    st.session_state.df_train = conn.execute("SELECT * FROM df_train").fetchdf()