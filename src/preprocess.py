# src/preprocess.py

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(path='data/emails.csv'):
    df = pd.read_csv(path, encoding='latin-1')[['label', 'text']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df
