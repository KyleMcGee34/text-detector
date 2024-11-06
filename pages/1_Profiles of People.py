import streamlit as st
import subprocess
import os
import csv
import re
from analyze import analyze_text, processNewDataWithLabels
import pandas as pd

st.set_page_config(
    page_title="Text Detector"
)

st.title('Profiles of People Detector Model')

tab1, tab2 = st.tabs(["Single Text Evaluation", "Multiple Text Evaluation"])
with tab1:
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
    text = st.text_area("Text to analyze")

    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    if 'save' not in st.session_state:
        st.session_state.save = 'No'
    if 'correct' not in st.session_state:
        st.session_state.correct = 'No'
    csv_file_path = 'data.csv'

    def set_state(i):
        st.session_state.stage = i

    if st.session_state.stage == 0:
        if st.session_state.save == 'Yes':
            if text and st.session_state.correct and st.session_state.save:
                st.cache_data.clear()
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    # Write the row of data
                    save_text = re.sub(r'\n+', ' ', text)
                    writer.writerow([save_text, st.session_state.correct, st.session_state.final_label])
                st.session_state.save = 'No'
                st.session_state.correct = 'No'
        st.button('Analyze text', on_click=set_state, args=[1])

    if st.session_state.stage >= 1:
        final_label = analyze_text(text, 'Models/ProfileTextDetector/Tokenizer.json', 'Models/ProfileTextDetector/Parms.json', 'Models/ProfileTextDetector/Model.keras')
        st.session_state.final_label = final_label
        st.write(f"The text you provided is assumed to be :blue-background[{final_label}]")
        correct = st.radio('Was the models prediction correct?',
                        ['Yes','No','Not Sure'],
                        index=None)
        if correct in ['Yes','No','Not Sure']:
            set_state(2)

    if st.session_state.stage >= 2:
        save = st.radio('Should the text you provided be saved for future model creation?',
                        ['Yes','No'],
                        index=None)
        if save in ['Yes','No']:
            set_state(3)

    if st.session_state.stage >= 3:
        st.write('Thank you!')
        st.session_state.save = save
        st.session_state.correct = correct
        if st.session_state.save == 'Yes':
            st.button('Submit', on_click=set_state, args=[0])
        if st.session_state.save == 'No':
            st.button('Clear Selections', on_click=set_state, args=[0])

with tab2:
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

    else:
        st.write("Please upload a CSV file.")

    if st.button('Analyze File'):
        processNewDataWithLabels(text_column, target_column, predict_df=df,
                                 tokenizer_path='Models/ProfileTextDetector/Tokenizer.json',
                                 parm_path='Models/ProfileTextDetector/Parms.json',
                                 model_path='Models/ProfileTextDetector/Model.keras')