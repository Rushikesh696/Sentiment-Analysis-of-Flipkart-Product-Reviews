# 📦 Sentiment Analysis of Flipkart Product Reviews (YONEX MAVIS 350)

This project focuses on building an end-to-end **sentiment analysis system** for classifying real-time Flipkart product reviews as **positive or negative**, with a focus on the *YONEX MAVIS 350 Nylon Shuttle*. The solution includes **text preprocessing**, **feature extraction**, **model training**, **evaluation**, and an interactive **Streamlit web application** for user inference.

---

## ❓ Problem Statement

With thousands of customer reviews generated on e-commerce platforms daily, it becomes challenging for sellers and buyers to manually analyze customer sentiment.  
The goal of this project is to:
- Automatically classify customer reviews as **positive or negative**.
- Identify patterns and common issues from **negative feedback** to improve product offerings.
- Enable a real-time, easy-to-use interface for sentiment detection through a web application.

---

## 🧠 Objective

- Classify customer reviews as **positive or negative**.
- Gain actionable insights from customer sentiments.
- Build an interactive Streamlit app for real-time sentiment prediction.

---

## 📂 Dataset

- **Source**: Real-time scraped data from Flipkart (provided dataset).
- **Product**: YONEX MAVIS 350 Nylon Shuttle
- **Total Reviews**: 8,518
- **Features**:  
  - Reviewer Name  
  - Rating  
  - Review Title  
  - Review Text  
  - Place of Review  
  - Date of Review  
  - Up Votes  
  - Down Votes  

---

## 🧹 Data Preprocessing

- Removed special characters, punctuation, and stopwords.
- Performed **lemmatization** to normalize text.
- Converted review text into numerical vectors using:
  - **TF-IDF**
  - **Word2Vec**
  - (Optional) **BERT embeddings**

---

## 🏗️ Modeling Approach

- Evaluated various machine learning and deep learning models:
  - Logistic Regression, Naive Bayes, SVM
  - **Random Forest** (Best)
  - LSTM (future scope)
- **Evaluation Metric**: F1-Score
- **Best Model**: ✅ **Random Forest Classifier** (highest F1-score)

---

## 🚀 Streamlit Web App

Developed a user-friendly **Streamlit app** that:
- Takes a product review as input
- Outputs real-time sentiment prediction
- Allows interactive testing of the trained model

---

## ☁️ Deployment *(Planned)*

The solution is ready for deployment on:
- **AWS EC2** instance *(planned)*
- Can also be deployed via Render or Hugging Face Spaces for public access

---

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, NLTK, Matplotlib
- **NLP Techniques**: TF-IDF, Word2Vec, BERT (optional)
- **Modeling**: Random Forest, Logistic Regression
- **App Development**: Streamlit
- **Deployment Target**: AWS EC2 (planned)

---

## 📁 Project Structure


---

## ✅ Conclusion

This project successfully demonstrates how **machine learning and NLP techniques** can be leveraged to extract meaningful insights from e-commerce product reviews.  
The **Random Forest** classifier achieved the best performance, and the entire solution was integrated into a **Streamlit web app** for real-time sentiment analysis.  
Deployment on **AWS EC2** remains a planned enhancement to make the solution accessible publicly.  

The project highlights skills in **data preprocessing, feature engineering, model evaluation, app development**, and **cloud readiness**—making it highly relevant for real-world applications in e-commerce and customer experience analytics.

---
 

