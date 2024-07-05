from nltk.corpus import stopwords
import nltk
import re
import tensorflow as tf
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text by replacing numerical values with a special token
def replace_numerical_values(text, token='<NUM>'):
    # Regular expression to match numerical values
    numerical_pattern = r'\b\d+\b'  # Matches one or more digits surrounded by word boundaries
    
    # Replace numerical values with the specified token
    preprocessed_text = re.sub(numerical_pattern, token, text)
    
    return preprocessed_text

# Function to lowercase all text
def lowercase_text(text):
    # Split the text into words
    words = text.split()
    
    # Lowercase each word and join them back into a string
    lowercased_text = ' '.join(word.lower() for word in words)
    
    return lowercased_text

@st.cache_resource
def get_model(path):
    loadedModel = tf.keras.models.load_model(path)
    return loadedModel

@st.cache_resource
def load_tokenizer(path):
    with open(path) as f: 
        data = json.load(f) 
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    return tokenizer

@st.cache_resource
def load_parms(path):
    with open(path, 'r') as json_file:
        parm_dict = json.load(json_file)
    return parm_dict


def analyze_text(text):
    # Create dataframe
    data = {'text': [text]}
    predict_df = pd.DataFrame(data)

    #Rremove stop words
    predict_df['text'] = predict_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Convert numbers to <NUM>
    predict_df['text'] = predict_df['text'].apply(replace_numerical_values)
    # Lower case everything
    predict_df['text'] = predict_df['text'].apply(lowercase_text)

    # Load tokenizer
    loaded_tokenizer = load_tokenizer('Models/ProfileTextDetector/Tokenizer.json')
    # Load parms
    loaded_parm_dict = load_parms('Models/ProfileTextDetector/Parms.json')
    # Load model
    loaded_model = get_model('Models/ProfileTextDetector/Model.keras')

    # Process text before prediction
    predict_texts = predict_df['text'].tolist()
    predict_sequences = loaded_tokenizer.texts_to_sequences(predict_texts)
    max_length = loaded_parm_dict['padding_value']
    predict_padded_sequences  = np.array(tf.keras.preprocessing.sequence.pad_sequences(predict_sequences, maxlen=max_length, padding='post'))
    # Prediction starts here
    predicted_probability = loaded_model.predict(predict_padded_sequences)
    # Convert prediction to int
    predicted_label = (predicted_probability > 0.5).astype(int)
    # Return final output
    scalar_value = predicted_label[0, 0]
    if scalar_value == 1:
        final_label = 'Human'
    if scalar_value == 0:
        final_label = 'Synthetic'

    return final_label