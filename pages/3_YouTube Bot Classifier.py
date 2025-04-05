import streamlit as st
from analyze_yt import analyze_text
import pandas as pd

st.set_page_config(
    page_title="YouTube Bot Classifier (Single)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>YouTube Bot Classifier (Single)</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Columns layout
col1, col2, col3 = st.columns([1.5, 1.5, 1])
with col3:
    with st.expander("Model Details", icon="🚨"):
        st.markdown("PLACE MODEL DETAILS HERE")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

# User input in col2
with col2:
    text = st.text_area("", height=200, label_visibility="collapsed")  # Empty string as the label to hide it
    col4, col5 = st.columns(2)
    with col4:
        timestamp = st.text_input("Enter Timestamp (Ex: 2024-10-08T22:46:34Z)",
                                       value="2024-10-08T22:46:34Z")
    with col5:
        date = st.text_input("Enter Date (Ex: 2024-10-08T22:46:34Z)",
                                       value="2024-10-08T22:46:34Z")
    col6, col7 = st.columns(2)
    with col6:
        username = st.text_input("Enter Username (Ex: @GrahamStephan)",
                                       value="@GrahamStephan")
    with col7:
        videoId = st.text_input("Enter VideoID (Ex: @GrahamStephan)",
                                       value="nGF4-8gRyn4")
    if st.session_state.stage == 0:
        st.button('Analyze text', on_click=set_state, args=[1])

with col1:
    if st.session_state.stage >= 1:
        st.markdown(
            """
            <div style="text-align: center;">
                <h2>Results</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("Below is a dataframe that can be used as inputs into an already trained model")
        final_label = analyze_text(text, timestamp, date, username, videoId)
        # st.markdown(final_label.loc[0, 'text'])
        st.dataframe(final_label)
        st.button('Reset', on_click=set_state, args=[0])
