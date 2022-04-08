
import tweepy
import time
import os
import csv

import json

import pandas as pd
import re



# def load_tweet():

account_list=["yahoo_marketing","Morisawa_JP"]

CK="KHR2ppJbJbDDpw767QGrriPkM"
CS="z8yHcnicE0LXkFSjxDgxawZ864KKycTpVXlJg6nRmvhP3xNvuw"
AT="1263612076586459136-bsVqLsh7D7mmobXt1CvIG5q8Z16uq8"
AS="scrw0VnsbBw9gU9VpEC2fdrFJoik5PTJLOtHtzMYUmVYZ"

auth = tweepy.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)

api=tweepy.API(auth)


num=2000
count=50
tweet_data=[]

for i in range(len(account_list)):
    tweets=tweepy.Cursor(api.user_timeline,
        id=account_list[i],
        include_rts=False,
        exclude_replies=True,
        tweet_mode='extended').items(num)
    print(i)
    # print(tweets,"tweets")
    # print(dir(tweets),"tweets")
    # users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
    # for users_loc in users_locs:
    #     print(users_loc,"users_loc")
# tweets=[[tweet.user.name]for tweet in tweets]

# with open("tweets.csv","w",newline="",encoding='utf-8') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerow(["id","created_at","text","fav","RT"])
#     writer.writerows(tweet_data)

    for tweet in tweets:

        # print(tweet.text,"str")
        # print(str(tweet))
        # print(tweet.favorite_count)
        if tweet.favorite_count>=count & tweet.retweet_count >= count:
            tweet_data.append([
                            tweet.user.name,
                            tweet.user.screen_name,
                            tweet.retweet_count,
                            tweet.favorite_count,
                            tweet.created_at.strftime('%Y-%m-%d'),
                            tweet.full_text.replace('\n','')

            ])
    # print(tweet)



# trainデータの表示

print(tweet_data)
df=pd.DataFrame(tweet_data,
                columns=['account_name','user_screen_name','rt','fav','date','text'])


# test data の読み込み
t_account_list=["kobelcokenki","ishiimark_sign","Maruyasu_1955"]

tweet_test_data=[]

for i in range(len(t_account_list)):
    tweets=tweepy.Cursor(api.user_timeline,
        id=t_account_list[i],
        include_rts=False,
        exclude_replies=True,
        tweet_mode='extended').items(num)
    print(i)
    # print(tweets,"tweets")
    # print(dir(tweets),"tweets")
    # users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
    # for users_loc in users_locs:
    #     print(users_loc,"users_loc")
# tweets=[[tweet.user.name]for tweet in tweets]

# with open("tweets.csv","w",newline="",encoding='utf-8') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerow(["id","created_at","text","fav","RT"])
#     writer.writerows(tweet_data)

    for tweet in tweets:

        # print(tweet.text,"str")
        # print(str(tweet))
        # print(tweet.favorite_count)
        if tweet.favorite_count>=count & tweet.retweet_count >= count:
            tweet_test_data.append([
                            tweet.user.name,
                            tweet.user.screen_name,
                            tweet.retweet_count,
                            tweet.favorite_count,
                            tweet.created_at.strftime('%Y-%m-%d'),
                            tweet.full_text.replace('\n','')

            ])

df_test=pd.DataFrame(tweet_test_data,
                columns=['account_name','user_screen_name','rt','fav','date','text'])
                
tweets_train=df["text"]
tweets_test=df_test["text"]

df.to_csv("csv/tweet_train_data.csv")
df_test.to_csv("csv/tweet_test_data.csv")



