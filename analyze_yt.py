import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

def analyze_text(text, timestamp, date, username, videoId):
    # Preprocess YouTube Bot Single Text
    cleaned_text = re.sub("[^a-zA-Z]", " ", text) # we are only going to keep a-z A-Z letters
    cleaned_text = cleaned_text.lower() # converts to lowercase for consistency
    cleaned_text = cleaned_text.split() # splits every word by space. we now have individual words

    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and apply stemming
    clean_comments = []
    for word in cleaned_text:
        if word not in stop_words:
            stemmed_word = ps.stem(word)
            clean_comments.append(stemmed_word)

    # Join back into one string
    final_cleaned_text = " ".join(clean_comments)

    data = {'Text': [final_cleaned_text],
            'Timestamp': [timestamp],
            'Date': [date],
            'Username': [username],
            'VideoId': [videoId]}
    predict_df = pd.DataFrame(data)

    return predict_df