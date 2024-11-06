import streamlit as st

st.set_page_config(
    page_title="Text Detector"
)

st.write("# Welcome to the Text Detector Main Landing Page! 👋")

st.sidebar.success("Select the typ of text you would like to evaluate above.")

st.markdown(
    """
    On the left hand side of the screen, you will see an option to select which type of 
    text you would like to evaluate. Please choose from the options.
"""
)