from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from jcopml.utils import load_model
from nltk.tokenize import word_tokenize
from static.src.predict import sentiment_prediction, document_prediction
from static.src.functions import train_bow, train_tfidf, STOPWORDS


app = Flask(__name__)
# model = load_model("model/indonesian_general_election_sgd_tfidf.pkl")
model = load('model/sentiment_prediction.joblib')
doc_finder_tfidf, doc_finder_tfidf_matrix = train_tfidf(
    "data/bank_central_asia_news.csv", 'Hit Sentence', stopwords=STOPWORDS)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/sentiment_checker", methods=["GET", 'POST'])
def sentiment_checker():
    if request.method == "GET":
        return render_template('sentiment-checker.html')
    elif request.method == "POST":
        pred = sentiment_prediction(model)
        return render_template('sentiment-checker-result.html', prediction=pred)


@app.route("/document-finder", methods=["GET", 'POST'])
def document_finder():
    if request.method == "GET":
        return render_template('document-finder.html')
    elif request.method == "POST":
        query = request.form['text']
        prediction = document_prediction(
            query, "data/bank_central_asia_news.csv", 'Hit Sentence', doc_finder_tfidf, doc_finder_tfidf_matrix)
        return render_template('document-finder-result.html', document_list=prediction)


if __name__ == '__main__':
    app.run
