import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import joblib  # for loading .pkl
import torch
import nltk

vectorizer_path = 'Models/YouTubeCommentDetector/vectorizer.pkl'
model_path = 'Models/YouTubeCommentDetector/full_model.pt'
# Load vectorizer
vectorizer = joblib.load(vectorizer_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full model
model = torch.load(model_path, map_location=device)
model.eval()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def analyze_text(text, timestamp, date, username, videoId):
    # Preprocess YouTube Bot Single Text
    cleaned_text = re.sub("[^a-zA-Z]", " ", text) # we are only going to keep a-z A-Z letters
    cleaned_text = cleaned_text.lower() # converts to lowercase for consistency
    cleaned_text = cleaned_text.split() # splits every word by space. we now have individual words

    ps = PorterStemmer()

    # Remove stopwords and apply stemming
    clean_comments = []
    for word in cleaned_text:
        if word not in stop_words:
            stemmed_word = ps.stem(word)
            clean_comments.append(stemmed_word)

    # Join back into one string
    final_cleaned_text = " ".join(clean_comments)

    # Vectorize the cleaned text
    X_vectorized = vectorizer.transform([final_cleaned_text])

    # Convert to tensor if necessary
    X_tensor = torch.tensor(X_vectorized.toarray(), dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(X_tensor)                      # Log probabilities
        predicted_class = torch.argmax(output, dim=1) # 0 or 1
        prediction = predicted_class.cpu().numpy()    # Convert to numpy if needed

    final_prediction = prediction[0]

    if final_prediction == 1:
        final_label = 'Human'
    if final_prediction == 0:
        final_label = 'Synthetic'

    data = {'Text': [final_cleaned_text],
            'Timestamp': [timestamp],
            'Date': [date],
            'Username': [username],
            'VideoId': [videoId]}
    predict_df = pd.DataFrame(data)


    return final_label