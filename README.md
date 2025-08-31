📰 News Classifier - AI-Powered Fake News Detection
===================================================

News Classifier is an AI-powered fake news detection system built with Python, spaCy, 
and Scikit-learn. It classifies news articles as Fake or Real using advanced NLP embeddings 
and machine learning algorithms.



About The Project
-----------------
This project is a machine learning classifier that predicts whether a news article is 
Fake or Real. It uses spaCy’s large English embeddings (en_core_web_lg) to extract 
vector representations of news text, and applies Naive Bayes and KNN classifiers for detection.  

It can be extended for real-time news verification systems or media authenticity checks.


Key Features
------------
- Fake vs Real News Detection
- Embeddings with spaCy for semantic text understanding
- Multiple ML Models: Naive Bayes & KNN
- Evaluation Metrics: Confusion Matrix & Classification Report
- Scalable and easily extendable to more datasets and models


Tech Stack
----------
- Python .......... Core programming language for ML and data processing
- spaCy ........... Used for text embeddings via pre-trained NLP models (en_core_web_lg)
- Scikit-learn .... Provides ML models (Naive Bayes, KNN) and evaluation metrics
- Pandas .......... Data handling and preprocessing
- NumPy ........... Numerical operations and array manipulations
- Jupyter ......... Interactive development and testing environment


Getting Started
---------------
1. Clone the repository
   git clone https://github.com/your-username/news-classifier.git
   cd news-classifier

2. Install dependencies
   pip install -r requirements.txt

3. Download spaCy model
   python -m spacy download en_core_web_lg

4. Run Jupyter Notebook
   jupyter notebook


Project Structure
-----------------
.
├── data/                # Dataset (Fake_Real_Data.csv)
├── notebooks/           # Jupyter notebooks for experiments
├── README.txt           # Project documentation



Results
-------
- Naive Bayes Accuracy: 87%
- KNN Accuracy: 82%
- Precision / Recall / F1: See classification_report.txt for details

Dataset
-------
- Source: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news)
- Total Samples: 44,898
   - Fake: 23,481
   - Real: 21,417

Future Improvements
-------------------
- Add more ML models (Logistic Regression, Random Forest)
- Experiment with Deep Learning (BERT, RoBERTa)
- Build a Streamlit/Flask web app for live news detection
- Deploy model as an API for integration with external systems

Contributors
------------
- Vikas Gautam (https://github.com/vikasgautam2003)


