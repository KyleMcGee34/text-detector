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
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import base64
import plotly.graph_objects as go

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

def create_circular_chart(label, value, color):
    """Creates a smaller circular chart using Plotly."""
    fig = go.Figure(
        go.Pie(
            values=[value, 100 - value],
            labels=["", ""],
            hole=0.90,  # Larger hole for a smaller overall size
            textinfo="none",
            marker=dict(colors=[color, "#E8E8E8"]),
            sort=False
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),  # Tight margins to make the chart smaller
        width=100,  # Set a fixed width for the chart
        height=100,  # Set a fixed height for the chart
        annotations=[
            dict(
                text=f"{value}%<br>{label}",
                x=0.5,
                y=0.5,
                font=dict(size=13, color=color),  # Smaller font size for percentage
                showarrow=False
            )
        ]
    )
    return fig
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
def analyze_text(text, tokenizer_path, parm_path, model_path):
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
    loaded_tokenizer = load_tokenizer(tokenizer_path)
    # Load parms
    loaded_parm_dict = load_parms(parm_path)
    # Load model
    loaded_model = get_model(model_path)

    # Process text before prediction
    predict_texts = predict_df['text'].tolist()
    predict_sequences = loaded_tokenizer.texts_to_sequences(predict_texts)
    max_length = loaded_parm_dict['padding_value']
    predict_padded_sequences  = np.array(tf.keras.preprocessing.sequence.pad_sequences(predict_sequences, maxlen=max_length, padding='post'))
    # Prediction starts here
    predicted_probability = loaded_model.predict(predict_padded_sequences)
    human_likelihood = round(predicted_probability[0, 0] * 100, 1)
    synthetic_likelihood= round((1 - predicted_probability[0, 0]) * 100, 1)
    # Use columns to display the charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_circular_chart("Human", human_likelihood, "#1f77b4"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_circular_chart("Synthetic", synthetic_likelihood, "#ff7f0e"),
            use_container_width=True
        )
    # st.progress(text=f"Human Likelihood :red-background[{round(predicted_probability[0, 0] * 100, 1)}%]",value=int(predicted_probability[0, 0] * 100))
    # st.progress(text=f"Synthetic Likelihood :red-background[{round((1 - predicted_probability[0, 0]) * 100, 1)}%]",value=int((1 - predicted_probability[0, 0]) * 100))
    # Convert prediction to int
    predicted_label = (predicted_probability > 0.5).astype(int)
    # Return final output
    scalar_value = predicted_label[0, 0]
    if scalar_value == 1:
        final_label = 'Human'
    if scalar_value == 0:
        final_label = 'Synthetic'
    return final_label

def analyze_classification(text, tokenizer_path, model_path, max_length=512, threshold=0.90):
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
    loaded_tokenizer = load_tokenizer(tokenizer_path)
    # Load model
    loaded_model = get_model(model_path)

    # Convert text to sequence
    sequence = loaded_tokenizer.texts_to_sequences(predict_df['text'].tolist())
    
    # Pad the sequence
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Get prediction probabilities
    probabilities = loaded_model.predict(padded_sequence)
    
    # Get the highest probability and corresponding class
    max_prob = np.max(probabilities)
    predicted_class = np.argmax(probabilities, axis=1)[0]

    class_mapping = {
    0: 'CNN',
    1: 'neuters',
    2: 'csmonitor',
    3: 'dailymail',
    4: 'nytimes'
    }
    class_map = class_mapping.get(predicted_class, "Unknown source")  # Use mapping to return the site name

    st.plotly_chart(
        create_circular_chart(class_map, round(max_prob * 100, 1), "#1f77b4"),
        use_container_width=True
    )
    # Check if probability meets the threshold
    if max_prob >= threshold:
        return class_mapping.get(predicted_class, "Unknown source")  # Use mapping to return the site name
    else:
        return "We cannot accurately determine where this text came from"


def processNewDataWithLabels(textColumn, targetColumn, onlyOne=False, predict_df=None, plotROC=True, tokenizer_path=None, parm_path=None, model_path=None, file_name="output"):

    predict_df = predict_df.rename(columns={textColumn: "text"})
    predict_df = predict_df.rename(columns={targetColumn: "target"})

    # Make a copy of the original text column before any modifications
    predict_df['Text'] = predict_df['text'].copy()

    # Remove stop words
    predict_df['text'] = predict_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Convert numbers to <NUM>
    predict_df['text'] = predict_df['text'].apply(replace_numerical_values)
    # Lower case everything
    predict_df['text'] = predict_df['text'].apply(lowercase_text)

    predict_texts = predict_df['text'].tolist()
    predict_labels = np.array(predict_df['target'].tolist())

    # Tokenization
    loaded_tokenizer = load_tokenizer(tokenizer_path)
    predict_sequences = loaded_tokenizer.texts_to_sequences(predict_texts)

    # Padding
    loaded_parm_dict = load_parms(parm_path)
    max_length = loaded_parm_dict['padding_value']
    predict_padded_sequences  = np.array(tf.keras.preprocessing.sequence.pad_sequences(predict_sequences, maxlen=max_length, padding='post'))

    # Get the predicted probabilities for each class
    loaded_model = get_model(model_path)
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
        # print(predicted_probabilities[:5])  # Inspect first few rows
        # positive_class_probabilities = predicted_probabilities[:, 0]  # Use index 0 to extract probabilities for the positive class
        positive_class_probabilities = predicted_probabilities  # Use index 0 to extract probabilities for the positive class
        # Calculate the false positive rate (fpr) and true positive rate (tpr) for different thresholds
        fpr, tpr, thresholds = roc_curve(predict_labels, positive_class_probabilities)
        # for true_label, pred_prob in zip(predict_labels, predicted_probabilities):
            # print(f"True Label: {true_label}, Predicted Probability: {pred_prob}")

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
        plt.title('Receiver Operating Characteristic (ROC) Curve\nSynthetic vs Human Written Results')
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

        # Create a combined figure with both plots for the PDF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # fig.suptitle(f"{file_name}_analysis", fontsize=20)  # Add the main title at the top
        disp.plot(cmap=plt.cm.Blues, ax=ax1)
        ax1.set_xlabel("Predicted labels")
        ax1.set_ylabel("True labels")
        ax1.set_title("Confusion Matrix")

        ax2.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")

        # Save the combined figure to a PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            with PdfPages(tmpfile.name) as pdf:
                pdf.savefig(fig)  # Save the single-page combined figure
            pdf_path = tmpfile.name

        # Format the download name using the uploaded file name
        download_filename = f"{file_name}_analysis.pdf"

        # Provide download link for the PDF
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
            b64 = base64.b64encode(pdf_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">Download PDF of Plots</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Add the predicted labels to the original dataframe
        predict_df['Predicted'] = predicted_labels
        predict_df['Actual'] = predict_labels

        # Add a column indicating whether the prediction was correct
        predict_df['Result'] = np.where(predict_df['Predicted'] == predict_df['Actual'], 'Correct', 'Incorrect')
        # Convert 'actual' and 'predicted' columns from 0/1 to 'Synthetic'/'Human'
        predict_df['Actual'] = predict_df['Actual'].map({0: 'Synthetic', 1: 'Human'})
        predict_df['Predicted'] = predict_df['Predicted'].map({0: 'Synthetic', 1: 'Human'})
        st.write("Model Predictions and Results:")
        st.dataframe(predict_df[['Result', 'Actual', 'Predicted', 'Text']].sort_values(by='Result', ascending=False))
