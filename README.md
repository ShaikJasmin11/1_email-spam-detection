#  Email Spam Detection

>  RISE Internship Project 1 – Tamizhan Skills  
>  Built with Scikit-Learn, Streamlit, and TfidfVectorizer

A machine learning-based web app that classifies emails as **spam** or **ham** using a trained Naive Bayes model. This is the first project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a lightweight and accurate email spam classifier that:
- Cleans and processes raw text
- Uses TF-IDF for feature extraction
- Trains a Naive Bayes model to classify email messages
- Provides a user-friendly web interface using Streamlit

---

##  Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas / NumPy**
- **TF-IDF Vectorizer**
- **Naive Bayes Classifier**
- **Streamlit** (for UI)
- **Joblib** (to save models)

---

##  Project Structure

```bash
email-spam-detection/
├── app.py                  # Streamlit frontend
├── main.py                 # Model training script
├── requirements.txt        # All required packages
├── data/
│   └── spam.csv            # Raw dataset
├── models/
│   └── spam_model.pkl      # Trained spam classifier
│   └── tfidf_vectorizer.pkl
├── src/
│   └── preprocess.py       # Text preprocessing functions
├── .gitignore
└── README.md               # You're reading it 😉
```
---

## Dataset

- Source: Kaggle – SMS Spam Dataset
- Contains ~5,000 labeled emails (spam/ham)

---

## How to Run

- Step 1: Install Dependencies
  ``` bash
  pip install -r requirements.txt
  ```

- Step 2: Train the Model
  ``` bash
  python main.py
  ```

- Step 3: Launch the Web App
  ``` bash
  streamlit run app.py
  ```

  ---

## Model Performance

✅ Accuracy: ~92%

✅ Fast inference (Naive Bayes)

✅ Can be integrated with contact forms, email inboxes, or APIs

---

## Highlights

- Cleaned using custom regex and preprocessing
- TF-IDF vectorization for text encoding
- Streamlit interface with real-time spam prediction
- Production-ready structure with modular Python code

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world spam filtering problems.

Built by @ShaikJasmin11
