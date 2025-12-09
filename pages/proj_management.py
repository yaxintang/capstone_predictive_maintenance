import streamlit as st
import streamlit_pages.st_function as f
import streamlit.components.v1 as components


# Set configuration and sidebar navigation
st.set_page_config(
    page_title = "Predictive Maintenance - Project Management",
    page_icon=":gear:",
    layout="wide"
)

f.navigation()

################## CONTENT START

############ TITLE

# Set Title
st.title("Project Management")

###### SUBHEADER
st.write("")
st.subheader("Timeline")
st.markdown(
    """
            
"""
            )

st.components.v1.iframe("https://second-postage-a2a.notion.site/ebd/2ae62696334780eb92a8e8dd1068d1e9?v=2ae62696334780ea8ce9000c0169f273", 
                        width=1400, 
                        height=1000, 
                        scrolling=True, 
                        tab_index=None)