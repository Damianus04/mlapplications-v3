from flask import Flask, render_template, url_for, request
import pandas as pd
import nltk
from joblib import load
from static.src.sentiment_prediction import sentiment_prediction
from static.src.document_finder import train_bow, train_tfidf, STOPWORDS, document_prediction
from static.src.twitter_sentiment_prediction import get_tweets, predict_sentiment, integrate_sentiment_and_df, text_preprocessing


app = Flask(__name__)
# model = load_model("model/indonesian_general_election_sgd_tfidf.pkl") #works
# model = load_model("model/logreg_bow_pipeline_sentiment_checker.pkl")  # works
# model = load_model("model/sgd_tfidf_wo_pipeline_sentiment_checker.pkl")
# model = load("model/logreg_tfidf.joblib")  # works
# model = load('model/sentiment_prediction.joblib')  # works
model = load('model/rand_search_logreg_hyper_tfidf_sklearn24.joblib')  # works
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

# SOCIAL SENTIMENT CHECKER


@app.route("/social-sentiment-checker", methods=["GET", 'POST'])
def social_sentiment_checker():
    if request.method == "GET":
        return render_template('social-sentiment-checker.html')
    # elif request.method == "POST":
    #     pred = sentiment_prediction(model)
    #     return render_template('twitter-sentiment-analysis-result.html', prediction=pred)

    if request.method == "GET":
        return render_template('social-sentiment-checker.html')
    elif request.method == "POST":
        # Get Tweet
        text_query = request.form['text']
        twitter_data = get_tweets(text_query)

        # add text tweet stats
        total_mentions = len(twitter_data)
        if total_mentions == 0 or text_query == " ":
            average_mentions = 0
        else:
            average_mentions = round(total_mentions/7)

        try:
            # add detailed datetime features
            twitter_data['date'] = twitter_data.created_at.apply(
                lambda x: x.date())
            twitter_data['day'] = twitter_data.created_at.apply(
                lambda x: x.day_name())
            twitter_data['month'] = twitter_data.created_at.apply(
                lambda x: x.month_name())
            twitter_data['year'] = twitter_data.created_at.apply(
                lambda x: x.year)
            twitter_data['time1'] = twitter_data.created_at.apply(
                lambda x: x.to_period('H').strftime('%d-%b-%y'))
            twitter_data['time2'] = twitter_data.created_at.apply(
                lambda x: x.to_period('H').strftime('%d-%b-%y %H:%M'))

            time1 = twitter_data[['time1', 'tweet_text']].groupby(
                ['time1'], as_index=False).count()
            time2 = twitter_data[['time2', 'tweet_text']].groupby(
                ['time2'], as_index=False).count()

            tweet_legend = "conversations"
            # choose whether to use time1 (day & hour) or time2 (hour)
            if len(time1.time1) > 1:
                tweet_time_label = list(time1.time1)
                tweet_count_values = list(time1.tweet_text)
            else:
                tweet_time_label = list(time2.time2)
                tweet_count_values = list(time2.tweet_text)

            # Preprocessing 'tweet_text'
            twitter_data['tweet_text_preprocessed'] = twitter_data['tweet_text'].apply(
                lambda x: text_preprocessing(x)
            )

            # # potential reach data
            reach_data = twitter_data[['screen_name', 'followers']].head(
                10).sort_values(by='followers', ascending=False)
            reach_data_screen_name = list(reach_data.screen_name)
            reach_data_followers = list(reach_data.followers)

            # # word distribution
            word_freq_dist_dict = []
            for i in twitter_data.tweet_text_preprocessed:
                word_freq_dist_dict.extend(i.split(' '))

            word_freq_dist = nltk.FreqDist(word_freq_dist_dict)
            top10words = word_freq_dist
            top10words = word_freq_dist.most_common(20)
            words = []
            words_frequency = []
            for i in top10words:
                words.append(i[0])
                words_frequency.append(i[1])

            # # location distribution
            top10locations = pd.DataFrame(
                twitter_data['location'].value_counts())[:20]
            locations = top10locations.index
            locations_frequency = [i for i in top10locations.location]

            # Predict Sentiment
            model_prediction = model
            text_list = twitter_data
            colname = "tweet_text_preprocessed"

            sentiment_tweet_data = predict_sentiment(
                model_prediction, text_list, colname)

            # Integrate Sentiment into Data
            get_tweet_result_data = twitter_data
            sentiment_colname = "sentiment"
            sentiment_result = sentiment_tweet_data
            df = integrate_sentiment_and_df(
                get_tweet_result_data, sentiment_colname, sentiment_result)

            sentiment_chart = df[['sentiment', 'tweet_text']].groupby(
                ['sentiment'], as_index=False).count()
            tweet_sentiment_label = list(sentiment_chart.sentiment)
            tweet_sentiment_values = list(sentiment_chart.tweet_text)
        except:
            tweet_legend = "conversations"
            tweet_time_label = ['None']
            tweet_count_values = [0]
            tweet_sentiment_label = ['None']
            tweet_sentiment_values = [0]
            reach_data_screen_name = ['None']
            reach_data_followers = [0]
            words = ['None']
            words_frequency = [0]
            locations = ['None']
            locations_frequency = [0]

        return render_template('social-sentiment-checker.html', tweet_data=df, data=text_query, text_query=text_query,
                               total_mentions=total_mentions, average_mentions=average_mentions,
                               #    legend=legend, labels=labels, values=values,
                               tweet_time_label=tweet_time_label, tweet_count_values=tweet_count_values, tweet_legend=tweet_legend,
                               tweet_sentiment_label=tweet_sentiment_label, tweet_sentiment_values=tweet_sentiment_values,
                               reach_data_screen_name=reach_data_screen_name, reach_data_followers=reach_data_followers,
                               words=words, words_frequency=words_frequency, locations=locations, locations_frequency=locations_frequency
                               )


if __name__ == '__main__':
    app.run
