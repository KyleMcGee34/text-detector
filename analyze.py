from nltk.corpus import stopwords
import nltk
import re
import tensorflow as tf
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

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
\
@st.cache_resource
def load_parms(path):
    with open(path, 'r') as json_file:
        parm_dict = json.load(json_file)
    return parm_dict

@st.cache_data
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
    st.progress(text=f"Human Likelihood :red-background[{round(predicted_probability[0, 0] * 100, 1)}%]",value=int(predicted_probability[0, 0] * 100))
    st.progress(text=f"Synthetic Likelihood :red-background[{round((1 - predicted_probability[0, 0]) * 100, 1)}%]",value=int((1 - predicted_probability[0, 0]) * 100))
    # Convert prediction to int
    predicted_label = (predicted_probability > 0.5).astype(int)
    # Return final output
    scalar_value = predicted_label[0, 0]
    if scalar_value == 1:
        final_label = 'Human'
    if scalar_value == 0:
        final_label = 'Synthetic'
    return final_label

def processNewDataWithLabels(textColumn, targetColumn, onlyOne=False, predict_df=None, plotROC=True):

    predict_df = predict_df.rename(columns={textColumn: "text"})
    predict_df = predict_df.rename(columns={targetColumn: "target"})

    # Remove stop words
    predict_df['text'] = predict_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Convert numbers to <NUM>
    predict_df['text'] = predict_df['text'].apply(replace_numerical_values)
    # Lower case everything
    predict_df['text'] = predict_df['text'].apply(lowercase_text)

    predict_texts = predict_df['text'].tolist()
    predict_labels = np.array(predict_df['target'].tolist())

    # Tokenization
    loaded_tokenizer = load_tokenizer('Models/ProfileTextDetector/Tokenizer.json')
    predict_sequences = loaded_tokenizer.texts_to_sequences(predict_texts)

    # Padding
    loaded_parm_dict = load_parms('Models/ProfileTextDetector/Parms.json')
    max_length = loaded_parm_dict['padding_value']
    predict_padded_sequences  = np.array(tf.keras.preprocessing.sequence.pad_sequences(predict_sequences, maxlen=max_length, padding='post'))

    # Get the predicted probabilities for each class
    loaded_model = get_model('Models/ProfileTextDetector/Model.keras')
    predicted_probabilities = loaded_model.predict(predict_padded_sequences)

    # Convert the predicted probabilities to class labels (0 or 1)
    predicted_labels = (predicted_probabilities > 0.5).astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(predict_labels, predicted_labels)

    accuracy = accuracy_score(predict_labels, predicted_labels)
    st.write(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    # Plot the confusion matrix
    if not onlyOne:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        # plt.show()
        cmPlot = plt.gcf()

    if plotROC:
        # Extract the probabilities for the positive class
        positive_class_probabilities = predicted_probabilities[:, 0]  # Use index 0 to extract probabilities for the positive class

        # Calculate the false positive rate (fpr) and true positive rate (tpr) for different thresholds
        fpr, tpr, thresholds = roc_curve(predict_labels, positive_class_probabilities)

        # Calculate the area under the ROC curve (AUC)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve\nHackathon Results')
        plt.legend(loc="lower right")
        rocPlot = plt.gcf()
        # plt.show()

        col1, col2 = st.columns(2)

        with col1:
            try:
                st.pyplot(cmPlot)
            except:
                st.write("Cannot calculate confusion matrix. Please ensure that the CSV file has at least one synthetic and human text")
        with col2:
            try:
                st.pyplot(rocPlot)
            except:
                st.write("Cannot calculate ROC plot. Please ensure that the CSV file has at least one synthetic and human text")