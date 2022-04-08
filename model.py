import time
import os
import csv

import json

import pandas as pd
from flask import Flask, render_template, request
import re
import unicodedata

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

#Mecabのインポート
# ! pip install mecab-python3 unidic-lite
import MeCab
from gensim.corpora.dictionary import Dictionary
# import corpora
from gensim.models import LdaModel
from collections import defaultdict

import math


    

def normalize(text):
        normalized_text = normalize_unicode(text)
        normalized_text = normalize_number(normalized_text)
        normalized_text = lower_text(normalized_text)
        return normalized_text


def lower_text(text):
    return text.lower()


def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text


def lemmatize_term(term, pos=None):
    if pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)


def normalize_number(text):
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text


def predict(user_tweets):

    # モデル実装

    # MeCabオブジェクトの生成
    mt = MeCab.Tagger('')
    mt.parse('')#バグ回避

    # トピック数の設定
    NUM_TOPICS = 3

    # トレーニングデータの読み込み
    # train_texts は二次元のリスト
    # テキストデータを一件ずつ分かち書き（名詞、動詞、形容詞に限定）して train_texts に格納するだけ
    train_texts = []
    df=pd.read_csv("csv/tweet_train_data.csv")
    tweets_train=df["text"]
    # print(tweets_train,"tweets_train")
    for line in tweets_train:
        text = []
        #nodeに解析結果を代入
        node = mt.parseToNode(line.strip())#line.strip()
        while node:
            fields = node.feature.split(",")
            if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                
                seikika=normalize(node.surface)
                seikika2=lower_text(seikika)
                seikika3=normalize_unicode(seikika2, form='NFKC')
                # seikika4=lemmatize_term(seikika3, pos=None)
                seikika5=normalize_number(seikika3)
                text.append(seikika5)
            node = node.next
        train_texts.append(text)
        #変更箇所
        
    # print(train_texts,"train_text")
    
    # モデル作成
    model_dictionary = Dictionary(train_texts)
    # print(dictionary)
    # model_dictionary.filter_extremes(no_below=2, no_above=0.1)
    corpus = [model_dictionary.doc2bow(texts) for texts in  train_texts]
    print(type(corpus), corpus)
    lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=model_dictionary, alpha=0.01,random_state=1)

    # テストデータ読み込み
    # test_texts は train_texts と同じフォーマット
    test_texts = []
    raw_test_texts = []
    df_test=pd.read_csv("csv/tweet_test_data.csv")
    tweets_test=df_test["text"]
    for line in tweets_test:
        text = []
        raw_test_texts.append(line.strip())
        node = mt.parseToNode(line.strip())
        while node:
            fields = node.feature.split(",")#,のところで区切る
        
            if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                seikika=normalize(node.surface)
                seikika2=lower_text(seikika)
                seikika3=normalize_unicode(seikika2, form='NFKC')
                seikika4=lemmatize_term(seikika3, pos=None)
                seikika5=normalize_number(seikika4)
                text.append(seikika5)
            node = node.next
        test_texts.append(text)
    
    # テストデータをモデルに掛ける
    score_by_topic = defaultdict(int)
    #既存の辞書を使用してコーパス作成
    #文書のベクトル化
    test_corpus = [model_dictionary.doc2bow(text) for text in test_texts]
    result=[]
    train_content=[]
    topic_id=[]
    topic_probability=[]
    topic_data=[]
    
    # クラスタリング結果を出力
    for unseen_doc, raw_train_text in zip(test_corpus, raw_test_texts):
        train_content.append(raw_train_text)
    
        for topic, score in lda[unseen_doc]:
            score_by_topic[int(topic)] = float(score)
        
        
        topic_probability.append(max(score_by_topic.values()))
        topic_id.append(max(score_by_topic, key=score_by_topic.get))
    
    topic_data.append([
                    train_content,
                    topic_id,
                    topic_probability
                    
    ])
    print(topic_data,"topic_data")



    #ユーザー入力読み込み
    test_texts_user = []
    raw_test_texts_user = []
    user_tweets=user_tweets
    print(user_tweets,"tweet_from_user")
    # text_user = []
    raw_test_texts_user.append(user_tweets)
    node = mt.parseToNode(user_tweets)
    print(raw_test_texts_user,"raw_test_texts_user")
    result_user=[]
    train_content_user=[]
    topic_id_user=[]
    topic_probability_user=[]
    topic_data_user=[]
    score_by_topic_user = defaultdict(int)
    
    # クラスタリング結果を出力
    for unseen_doc, raw_train_text in zip(test_corpus, raw_test_texts_user):
        train_content_user.append(raw_train_text)
        
        for topic, score in lda[unseen_doc]:
            score_by_topic_user[int(topic)] = float(score)
        topic_probability_user.append(max(score_by_topic_user.values()))
        topic_id_user.append(max(score_by_topic_user, key=score_by_topic_user.get))

    # topic_probability_user=math.floor(topic_probability_user)
        
    topic_data_user.append([
                    train_content_user,
                    topic_id_user,
                    topic_probability_user                       
    ])
    
    
    print(topic_data_user,"topic_data_user")
    print(topic_id_user,"topic_id_user")
    

    df_train_content=pd.DataFrame(train_content,
                                columns=['train_content'])
    df_topic_id=pd.DataFrame(topic_id,
                            columns=['topic_id'])
    df_topic_probability=pd.DataFrame(topic_probability,
                                    columns=['topic_probability'])
    df_concat=pd.concat([df_train_content,df_topic_id,df_topic_probability],axis=1)

    print(df_concat,"df_concat")
    buzz_id=list(df_concat['topic_id'].mode())
    buzz_topic=buzz_id[0]

    probability=score_by_topic_user[buzz_topic]
    print(probability,"probability")
    #if文
    print(topic_probability_user,"topic_probability_user")
    if topic_id_user==buzz_topic:
        return topic_probability_user
    else:
        return probability

if __name__ == "__main__":
    # user_tweets=input()
    # load_tweet()
    predict()