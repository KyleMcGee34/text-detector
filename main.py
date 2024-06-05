import streamlit as st
import subprocess
import os
from analyze import analyze_text

st.title('Text Detector Model')
text = st.text_area("Text to analyze")
analyze = st.button('Analyze text')
if analyze:
    # Call the function from the other script
    final_label = analyze_text(text)

    st.write("The text you provided is assumed to be ", final_label)