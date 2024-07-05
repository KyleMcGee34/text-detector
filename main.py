import streamlit as st
import subprocess
import os
import csv
import re
from analyze import analyze_text

st.title('Text Detector Model')
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
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write the row of data
                save_text = re.sub(r'\n+', ' ', text)
                writer.writerow([save_text, st.session_state.correct, st.session_state.final_label])
            st.session_state.save = 'No'
            st.session_state.correct = 'No'
    st.button('Analyze text', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    final_label = analyze_text(text)
    st.session_state.final_label = final_label
    # st.write(f"temprature: :blue[{temperature}]")
    st.write(f"The text you provided is assumed to be :blue-background[{final_label}]")
    # st.write("The text you provided is assumed to be --->", final_label, "<---")
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
    st.cache_data.clear()
    st.button('Start Over', on_click=set_state, args=[0])
