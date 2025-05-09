import streamlit as st
import subprocess
import os
import csv
import re
from analyze import analyze_text, processNewDataWithLabels, analyze_classification, analyze_classificationSynthetic
import pandas as pd

st.set_page_config(
    page_title="News Article Detector (Single)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>News Article Detector Model</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Columns layout
col1, col2, col3 = st.columns([1, 1.5, 1])
with col3:
    with st.expander("Model Details", icon="🚨"):
        data = [['2492', 'Synthetic'], ['2492', 'Human']]
        df = pd.DataFrame(data, columns=['Count', 'Type'])
        st.write('Training data counts:')
        st.dataframe(df,hide_index=True)
        st.write('''
            This model is trained on synthetic and human written news articles. 
            These were the models used in training: 
                
                - mistral-nemo:latest
                
                - phi3:14b 
                
                - hermes3:70b
                
                - gemma2:27b
                
                - llama3.1:70b
        ''')
        st.write('''
                 If the text is predicted to be human, a classification model will try to 
                 determine what website/news agency the text is most similar too. The websites used in
                 this model are:
                 
                 - [CNN Lite](https://lite.cnn.com/)

                 - [NYT](https://www.nytimes.com/)
                 
                 - [csmonitor](https://www.csmonitor.com/layout/set/text/textedition)

                 - [Daily Mail](https://www.dailymail.co.uk/textbased/channel-561/index.html)

                 - [neuters](https://neuters.de)
        ''')     
           
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'stage' not in st.session_state:
    st.session_state.stage = 0
if 'save' not in st.session_state:
    st.session_state.save = 'No'
if 'correct' not in st.session_state:
    st.session_state.correct = 'No'
csv_file_path = 'News_Articles_Data.csv'

def set_state(i):
    st.session_state.stage = i

# User input in col2
with col2:
    text = st.text_area("", height=200, label_visibility="collapsed", key="text")  # Empty string as the label to hide it
    if st.session_state.stage == 0:
        st.button('Analyze text', on_click=set_state, args=[1])

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
        final_label = analyze_text(text, 'Models/NewsArticleTextDetector/Tokenizer.json', 'Models/NewsArticleTextDetector/Parms.json', 'Models/NewsArticleTextDetector/Model.keras')
        st.session_state.final_label = final_label
        background_color = "blue" if final_label == "Human" else "red"
        st.markdown(
            f"""
            <div style="text-align: center;">
                <p>
                    The text you provided is assumed to be 
                    <span style="background-color: {background_color}; color: white; padding: 5px; border-radius: 5px;">
                        {final_label}
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if final_label == "Human":
            final_classification = analyze_classification(text, 'Models/NewsArticleTestClassifier/NewsArticlesClassification.json', 'Models/NewsArticleTestClassifier/NewsArticlesClassification.keras')
            if final_classification != "We cannot accurately determine where this text came from":
                final_classification_output = "This text most likely came from " + final_classification
            else:
                final_classification_output = final_classification + " using a 50% confidence level"
            st.markdown(
            f"""
            <div style="text-align: center;">
                <p>
                    {final_classification_output}
                </p>
            </div>
            """,
            unsafe_allow_html=True
            )
        if final_label == "Synthetic":
            final_classification = analyze_classificationSynthetic(text, 'Models/NewsArticleTestClassifier/NewsArticlesClassificationSyntheticModels.json', 'Models/NewsArticleTestClassifier/NewsArticlesClassificationSyntheticModels.keras')
            if final_classification != "We cannot accurately determine where this text came from":
                final_classification_output = "This text most likely came from " + final_classification
            else:
                final_classification_output = final_classification + " using a 50% confidence level"
            st.markdown(
            f"""
            <div style="text-align: center;">
                <p>
                    {final_classification_output}
                </p>
            </div>
            """,
            unsafe_allow_html=True
            )

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
            st.button('Submit', on_click=lambda: (set_state(0), setattr(st.session_state, "text", "")))
        if st.session_state.save == 'No':
            st.button('Clear Selections', on_click=lambda: (set_state(0), setattr(st.session_state, "text", "")))
