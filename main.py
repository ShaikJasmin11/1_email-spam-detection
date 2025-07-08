import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save_model():
    df = pd.read_csv('data/spam.csv', encoding='latin1')
    df = df[['v1', 'v2']]  # Keep only first two columns
    df.columns = ['label', 'text']  # Rename to cleaner names
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['text'].astype(str)
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("✅ Model & vectorizer saved.")

if __name__ == '__main__':
    train_and_save_model()
