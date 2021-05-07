from flask import Flask, render_template, url_for, request
import pandas as pd
from joblib import load
from static.src.sentiment_prediction import sentiment_prediction
from static.src.document_finder import train_bow, train_tfidf, STOPWORDS, document_prediction
from static.src.twitter_sentiment_prediction import get_tweets, predict_sentiment, integrate_sentiment_and_df


app = Flask(__name__)
# model = load_model("model/indonesian_general_election_sgd_tfidf.pkl") #works
# model = load_model("model/logreg_bow_pipeline_sentiment_checker.pkl")  # works
# model = load_model("model/sgd_tfidf_wo_pipeline_sentiment_checker.pkl")
# model = load("model/logreg_tfidf.joblib")  # works
model = load('model/sentiment_prediction.joblib')  # works
doc_finder_tfidf, doc_finder_tfidf_matrix = train_tfidf(
    "data/bank_central_asia_news.csv", 'Hit Sentence', stopwords=STOPWORDS)


@app.route("/")
def home():
    return render_template('index.html')

# SENTIMENT CHECKER


@app.route("/sentiment-checker", methods=["GET", 'POST'])
def sentiment_checker():
    if request.method == "GET":
        return render_template('sentiment-checker.html')
    elif request.method == "POST":
        text, pred = sentiment_prediction(model)
        return render_template('sentiment-checker.html', prediction=pred, data=text)

# DOCUMENT FINDER


@app.route("/document-finder", methods=["GET", 'POST'])
def document_finder():
    if request.method == "GET":
        return render_template('document-finder.html')
    elif request.method == "POST":
        query = request.form['text']
        prediction = document_prediction(
            query, "data/bank_central_asia_news.csv", 'Hit Sentence', doc_finder_tfidf, doc_finder_tfidf_matrix)
        return render_template('document-finder-result.html', document_list=prediction, data=query)

# TWITTER SENTIMENT CHECKER


@app.route("/twitter-sentiment-analysis", methods=["GET", 'POST'])
def twitter_sentiment_analysis():
    if request.method == "GET":
        return render_template('twitter-sentiment-analysis.html')
    # elif request.method == "POST":
    #     pred = sentiment_prediction(model)
    #     return render_template('twitter-sentiment-analysis-result.html', prediction=pred)

    if request.method == "GET":
        return render_template('twitter-sentiment-analysis.html')
    elif request.method == "POST":
        # Get Tweet
        text_query = request.form['text']
        tweet_data = get_tweets(text_query)

        # Predict Sentiment
        model_prediction = model
        text_list = tweet_data
        colname = "tweet_text"

        sentiment_tweet_data = predict_sentiment(
            model_prediction, text_list, colname)

        # Integrate Sentiment into Data
        get_tweet_result_data = tweet_data
        sentiment_colname = "sentiment"
        sentiment_result = sentiment_tweet_data
        df = integrate_sentiment_and_df(
            get_tweet_result_data, sentiment_colname, sentiment_result)

        return render_template('twitter-sentiment-analysis.html', prediction=df, data=text_query)


if __name__ == '__main__':
    app.run
