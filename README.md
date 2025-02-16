# Fake News Detection System

# Overview:

The Fake News Detection System is a machine learning-based project designed to classify news articles as either real or fake. This project utilizes Python, Jupyter Notebook for model development, and a front-end interface built with HTML and Tailwind CSS to provide an intuitive user experience.

# Features

Detects fake news using Natural Language Processing (NLP)

Uses machine learning models trained on labeled datasets

Provides a simple and interactive user interface

Built with Jupyter Notebook for easy experimentation and visualization

Frontend designed using HTML and Tailwind CSS

Tech Stack

Backend: Python (Flask for web framework)

Machine Learning: Scikit-learn, Pandas, NumPy, NLTK

Frontend: HTML, Tailwind CSS

IDE: Jupyter Notebook

Database: CSV/SQLite (for storing datasets)

Installation

Prerequisites

Ensure you have the following installed:

Python (>=3.8)

Jupyter Notebook

Virtual environment (optional but recommended)

Setup

# Clone this repository:

git clone https://github.com/Shivanshu63a/FakeNews_Detection.git
cd fake-news-detection

# Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies:

pip install -r requirements.txt

# Start Jupyter Notebook:

jupyter notebook

# Usage

Open FakeNewsDetection.ipynb in Jupyter Notebook.

Run each cell to train the model and evaluate performance.

Use the Flask server to serve the model and frontend interface:

python app.py

Open the browser and navigate to http://127.0.0.1:5000/.

Enter the news article in the input field and get the prediction.

# Dataset

The system uses datasets like Fake News Dataset from Kaggle, containing labeled real and fake news articles.

# Model Details

Vectorization: TF-IDF (Term Frequency - Inverse Document Frequency)

Classification Model: Logistic Regression / Naïve Bayes / Random Forest

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

Screenshots

(Include relevant UI screenshots here)

# Future Enhancements

Improve model accuracy with deep learning techniques (e.g., LSTMs, Transformers)

Implement real-time news scraping and classification

Add multilingual support for fake news detection

# Contributors

SHIVANSHU AGNIHOTRI

License

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments

Kaggle for dataset resources

Open-source libraries like Scikit-learn, Flask, and Tailwind CSS

