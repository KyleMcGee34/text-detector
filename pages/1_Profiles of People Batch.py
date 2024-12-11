import streamlit as st
import subprocess
import os
import csv
import re
from analyze import analyze_text, processNewDataWithLabels
import pandas as pd

st.set_page_config(
    page_title="Profiles of People Detector (Batch)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Profiles of People Detector Model (Batch)</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Columns layout
col1, col2, col3 = st.columns([1.5, 1.5, 1])
with col3:
    with st.expander("Model Details", icon="🚨"):
        data = [['1786', 'Synthetic'], ['1747', 'Human']]
        df = pd.DataFrame(data, columns=['Count', 'Type'])
        st.write('Training data counts:')
        st.dataframe(df,hide_index=True)
        st.write('''
            This model is trained on synthetic and human written profiles. 
            These were the models used in training: 
                
                - Llama-2-13B-GPTQ
                
                - EleutherAI_gpt-j-6B
                
                - facebook_galactica-6.7b
                
                - facebook_opt-6.7b
                
                - gpt2-xl
                
                - LoneStriker_dolphin-2.5-mixtral-8x7b-6.0bpw-h6-exl2-2
                
                - TB_Psyfighter-13B-GPTQ
                
                - TB_neural-chat-7B-v3-3-GPTQ
        ''')
with col2:
    st.markdown("When uploading a file please format it in the following way:")
    st.markdown("""
    - The file must be in CSV format
    - One column must contain the text you want to analyze
    - One column must contain the type of text:
        - Label synthetic text as 0
        - Label human text as 1
    - After uploading the CSV file, you will be prompted to select the columns for the text and type data
    """)
    uploaded_file = st.file_uploader("CSV File Upload", type="csv")
    if uploaded_file is not None:
        # Extract the filename without extension
        file_name = os.path.splitext(uploaded_file.name)[0]
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the dataframe
        st.write("Here is your data:")
        st.dataframe(df)

        # Get column names from the dataframe
        columns = df.columns.tolist()

        # Create dropdowns for selecting text and target columns
        textCol, targetCol = st.columns(2)
        with textCol:
            text_column = st.selectbox("Which column contains the text?", columns)
        with targetCol:
            target_column = st.selectbox("Which column contains the target?", columns)
        if st.button('Analyze File'):
            with col1:
                st.markdown(
                """
                <div style="text-align: center;">
                    <h2>Results</h2>
                </div>
                """, unsafe_allow_html=True)

                processNewDataWithLabels(text_column, target_column, predict_df=df,
                                            tokenizer_path='Models/ProfileTextDetector/Tokenizer.json',
                                            parm_path='Models/ProfileTextDetector/Parms.json',
                                            model_path='Models/ProfileTextDetector/Model.keras',
                                            file_name=file_name)

    else:
        st.write("Please upload a CSV file.")
# with col1:
#     if st.button('Analyze File'):
#         processNewDataWithLabels(text_column, target_column, predict_df=df,
#                                     tokenizer_path='Models/ProfileTextDetector/Tokenizer.json',
#                                     parm_path='Models/ProfileTextDetector/Parms.json',
#                                     model_path='Models/ProfileTextDetector/Model.keras',
#                                     file_name=file_name)