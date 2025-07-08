# app.py
import streamlit as st
import joblib
from src.preprocess import clean_text

# Load model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Page UI
st.set_page_config(page_title="Email Spam Classifier", layout="centered")
st.title("✉️ Email Spam Detector")
st.markdown("**Classify your message as Spam or Not Spam (Ham)**")

# Handle session state for prediction and input
if "prediction" not in st.session_state:
    st.session_state.prediction = ""
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

# RESET handler (before widget declaration)
if st.session_state.reset_triggered:
    st.session_state.user_input = ""
    st.session_state.prediction = ""
    st.session_state.reset_triggered = False
    st.rerun()

# Text area
user_input = st.text_area("Enter your Email/SMS message:", value=st.session_state.user_input, key="input_text")

# Action Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Predict"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            label = "🛑 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
            st.session_state.prediction = f"Prediction: **{label}**"
            st.session_state.user_input = user_input  # Save typed text
        else:
            st.warning("⚠️ Please enter a message.")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.reset_triggered = True
        st.rerun()

# Show prediction result
if st.session_state.prediction:
    st.success(st.session_state.prediction)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Built by <strong>Jasmin Shaik</strong></div>",
    unsafe_allow_html=True
)
