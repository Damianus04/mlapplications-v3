import pandas as pd
import tweepy
import time
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "VF9Xv4WxUBUBMrH2yntvW2DD0"
# api secret key
api_secret_key = "mzZU8SQUXHwYThBO0mtk3hkQG9JYhZ6byJoIjlwcBTagSG4Nkn"
# access token
access_token = "473787030-HUT2iebZltM6XjswhV7fnGpQerksZ5a7qaoyEJjp"
# access token secret
access_token_secret = "LkvliySmLd57Wt8LVVpy1nJ5jlRwaQo1wzfyh0epYzJ8d"

authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True)


def get_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 50
    try:
        # Pulling individual tweets from query
        for tweet in api.search(q=text_query, count=count):
            # print(tweet.text)
            # Adding to list that contains all tweets
            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})
        # return tweets_list
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)


def predict_sentiment(model, text_list, colname):
    sentiment_prediction = []
    target = model.predict(text_list[colname])
    for i, t in enumerate(target):
        if target[i] == 1:
            x = "positive"
        elif target[i] == -1:
            x = "negative"
        elif target[i] == 0:
            x = "neutral"

        sentiment_prediction.append(x)

    return sentiment_prediction


def integrate_sentiment_and_df(get_tweet_result_data, sentiment_colname, sentiment_result):
    get_tweet_result_data[sentiment_colname] = sentiment_result
    return get_tweet_result_data


# {{url_for('twitter_sentiment_analysis')}}
