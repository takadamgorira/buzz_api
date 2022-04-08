# サーバー側の処理を記述するためのPythonファイル
from flask import Flask, render_template, request
# from model import （model.pyの関数名）
from model import predict
import pandas as pd
# Flaskのインスタンス（Webアプリの本体）を作成
app = Flask(__name__)

# ルーティング
@app.route("/")
def input():
    return render_template('index.html')

@app.route("/input", methods=["GET", "POST"])
def upload_user_tweets():
    if request.method == "GET":
        return render_template('input.html')
    elif request.method == "POST":
        user_tweets = request.form.get('user_tweets')
        topic_probability_user=predict(user_tweets)
        topic_probability_user=round(topic_probability_user,2)
        topic_probability_user=topic_probability_user*100
        return render_template('result.html',topic_probability_user=topic_probability_user)

@app.route("/result")
def result():
    return render_template()

if __name__ == "__main__":
    app.run(debug=True)