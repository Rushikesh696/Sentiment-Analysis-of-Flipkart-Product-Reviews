# Sentiment Analysis of Flipkart Product Reviews (YONEX MAVIS 350)

This project focuses on building an end-to-end sentiment analysis system for classifying real-time Flipkart product reviews as positive or negative, with a focus on the YONEX MAVIS 350 Nylon Shuttle. The solution includes text preprocessing, feature extraction, model training, evaluation, and an interactive Streamlit web application for user inference.

---

### Problem Statement
With thousands of customer reviews generated on e-commerce platforms daily, it becomes challenging for sellers and buyers to manually analyze customer sentiment.
The goal of this project is to:

Automatically classify customer reviews as positive or negative.

Identify patterns and common issues from negative feedback to improve product offerings.

Enable a real-time, easy-to-use interface for sentiment detection through a web application.

---

### Objective
Classify customer reviews as positive or negative.

Gain actionable insights from customer sentiments.

Build an interactive Streamlit app for real-time sentiment prediction.

---

### Dataset
Source: Real-time scraped data from Flipkart (provided dataset).

Product: YONEX MAVIS 350 Nylon Shuttle

Total Reviews: 8,518

Features:

Reviewer Name

Rating

Review Title

Review Text

Place of Review

Date of Review

Up Votes

Down Votes

---

### Data Preprocessing
Removed special characters, punctuation, and stopwords.

Performed lemmatization to normalize text.

Converted review text into numerical vectors using:

TF-IDF

Word2Vec

---

### Modeling Approach
Evaluated various machine learning models:

Logistic Regression, Naive Bayes, SVM

Random Forest (Best performance among ML models)

Evaluation Metric: F1-Score

---
Deep Learning Comparison
To further evaluate the effectiveness of sequential neural networks in text classification, a separate module was added to compare:

🔹 Simple RNN

🔹 LSTM (Long Short-Term Memory)

🔹 GRU (Gated Recurrent Unit)

---

### Streamlit Web App
Developed a user-friendly Streamlit app that:

Takes a product review as input

Outputs real-time sentiment prediction

Allows interactive testing of the trained model

---

## ☁️ Deployment *(Planned)*

The solution is ready for deployment on:
- **AWS EC2** instance *(planned)*
- Can also be deployed via Render or Hugging Face Spaces for public access

---

### Tools & Technologies
Languages: Python

Libraries: Scikit-learn, Pandas, NumPy, NLTK, Matplotlib, TensorFlow, Keras

NLP Techniques: TF-IDF, Word2Vec

ML Models: Random Forest, Logistic Regression, Naive Bayes

DL Models: RNN, LSTM, GRU

App Development: Streamlit


---

### Conclusion
This project successfully demonstrates how machine learning and deep learning techniques can be leveraged to extract meaningful insights from e-commerce product reviews.
While the Random Forest classifier performed best among traditional models, LSTM emerged as the top performer in deep learning models.
The entire solution was integrated into a Streamlit web app for real-time sentiment analysis.

The project highlights skills in data preprocessing, NLP, ML/DL model building, evaluation, and app development, making it highly relevant for real-world applications in e-commerce and customer experience analytics.

---
 

