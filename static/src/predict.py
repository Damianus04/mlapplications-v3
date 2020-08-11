from flask import Flask, render_template, url_for, request
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd


def sentiment_prediction(model):
    text = request.form['text']
    data = [text]
    pred = model.predict(data)
    return pred


def document_prediction(query, filepath, col_name, tfidf, tfidf_matrix):
    df = pd.read_csv(filepath, encoding='iso-8859-1')
    vec = tfidf.transform([query])
    dist = cosine_distances(vec, tfidf_matrix)
    result_series = dist.argsort()[0, :10]
    result_list = result_series.tolist()
    result = df[col_name][result_list]
    document_list = result.tolist()
    return document_list
