import streamlit as st 
import joblib

# Title
st.title("Sentiment Analysis App")

# Load model and TF-IDF vectorizer
with open('raf_model.pkl', 'rb') as f:
    raf_model = joblib.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = joblib.load(f)

# Label mapping
label_map = {0: 'Negative', 1: 'Positive'}

# Input text
text = st.text_input("Enter the text for sentiment analysis")

# Predict button
if st.button("Predict"):
    if text:
        # Transform text using TF-IDF vectorizer
        transformed_text = tfidf.transform([text]).toarray()

        # Predict sentiment
        prediction = raf_model.predict(transformed_text)[0]

        # Display result
        sentiment = label_map[prediction]
        st.write(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.write("Please enter some text to analyze.")
