
Background
Twitter has become a wildly sprawling jungle of information—140 characters at a time. Somewhere between 350 million and 500 million tweets are estimated to be sent out per day. With such an explosion of data, on Twitter and elsewhere, it becomes more important than ever to tame it in some way, to concisely capture the essence of the data.

News Mood
In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: BBC, CBS, CNN, Fox, and New York times.

The first plot will be and/or feature the following:

Be a scatter plot of sentiments of the last 100 tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
Each plot point will reflect the compound sentiment of a tweet.
Sort each plot point by its relative timestamp.
The second plot will be a bar plot visualizing the overall sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

Pull last 100 tweets from each outlet.
Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
Export the data in the DataFrame into a CSV file.
Save PNG images for each plot.
As final considerations:

Use the Matplotlib and Seaborn libraries.
Include a written description of three observable trends based on the data.
Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
Include an exported markdown version of your Notebook called  README.md in your GitHub repository.



```python
#Dependencies
import os
import json
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import requests
from random import uniform
import tweepy
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#API keys
from config import consumer_key
from config import consumer_secret
from config import access_token
from config import access_token_secret

```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```

Twitter Handles:
(1) BBC - @BBC - https://twitter.com/BBC?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor
(2) CBS - @CBS - https://twitter.com/CBS?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor
(3) CNN - @CNN - https://twitter.com/CNN?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor
(4) Fox - @FoxNews - https://twitter.com/foxnews?lang=en
(5) New york Times - @nytimes -https://twitter.com/nytimes?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor


```python
#target users:
target_user = ["@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"]
```


```python
#lists
twt_handle = []
text = []
date = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
tweets_ago = []
```


```python
for user in target_user:
    count = 0
    for x in range(5):
#        count=0
        public_tweets = api.user_timeline(user,page=x)
        
        for tweet in public_tweets:
            count += 1
            twt_handle.append(user)
            text.append(tweet['text'])
            tweets_ago.append(count)
            date.append(tweet['created_at'])
            
            
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]

            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
```


```python
tweets = {
        "Tweets Ago" : tweets_ago,
        "Twitter Name": twt_handle,
        "Tweet" : text,
        "Date" : date,
        "Compound Score" : compound_list,
        "Positive Score" : positive_list,
        "Negative Score" : negative_list,
        "Neutral Score" : neutral_list
}
```


```python
tweets_df = pd.DataFrame(tweets, columns = ["Tweets Ago","Twitter Name", "Tweet", "Date", "Compound Score", "Positive Score", "Negative Score", "Neutral Score"])
tweets_df.to_csv('Tweets.csv')
tweets_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets Ago</th>
      <th>Twitter Name</th>
      <th>Tweet</th>
      <th>Date</th>
      <th>Compound Score</th>
      <th>Positive Score</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>@BBC</td>
      <td>Could this be an answer to global water shorta...</td>
      <td>Sun Mar 25 19:44:01 +0000 2018</td>
      <td>0.1280</td>
      <td>0.101</td>
      <td>0.077</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>@BBC</td>
      <td>Tonight, @regyates meets people whose lives ha...</td>
      <td>Sun Mar 25 19:15:06 +0000 2018</td>
      <td>-0.7506</td>
      <td>0.000</td>
      <td>0.286</td>
      <td>0.714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>@BBC</td>
      <td>Tonight, @mcgregor_ewan and @McgColin celebrat...</td>
      <td>Sun Mar 25 18:40:04 +0000 2018</td>
      <td>0.5719</td>
      <td>0.163</td>
      <td>0.000</td>
      <td>0.837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>@BBC</td>
      <td>The first ever statue of David Bowie has been ...</td>
      <td>Sun Mar 25 18:13:03 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>@BBC</td>
      <td>When you're enjoying being single and people j...</td>
      <td>Sun Mar 25 17:30:07 +0000 2018</td>
      <td>0.5267</td>
      <td>0.185</td>
      <td>0.000</td>
      <td>0.815</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>@BBC</td>
      <td>🇺🇸🏝🇬🇧 Welcome to Tangier Island, the tiny US i...</td>
      <td>Sun Mar 25 15:03:03 +0000 2018</td>
      <td>0.4588</td>
      <td>0.167</td>
      <td>0.000</td>
      <td>0.833</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>@BBC</td>
      <td>💬 We could listen to him speak all day. \n\n📽 ...</td>
      <td>Sun Mar 25 14:30:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>@BBC</td>
      <td>Predictions suggest a build-up of about 80,000...</td>
      <td>Sun Mar 25 14:09:03 +0000 2018</td>
      <td>0.1779</td>
      <td>0.091</td>
      <td>0.000</td>
      <td>0.909</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>@BBC</td>
      <td>👽✨🛸 @prattprattpratt stars as a happy-go-lucky...</td>
      <td>Sun Mar 25 13:03:04 +0000 2018</td>
      <td>0.5574</td>
      <td>0.184</td>
      <td>0.000</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>@BBC</td>
      <td>Weighing just 100g, a newborn panda is one 900...</td>
      <td>Sun Mar 25 12:03:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>@BBC</td>
      <td>🐟Meet SoFi - the soft robot fish developed by ...</td>
      <td>Sun Mar 25 10:30:04 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>@BBC</td>
      <td>Ever wondered what made you feel moody? It mig...</td>
      <td>Sun Mar 25 09:30:15 +0000 2018</td>
      <td>-0.3612</td>
      <td>0.000</td>
      <td>0.161</td>
      <td>0.839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>@BBC</td>
      <td>RT @BBCScotland: Up your brunch game with perf...</td>
      <td>Sun Mar 25 08:43:31 +0000 2018</td>
      <td>0.5719</td>
      <td>0.236</td>
      <td>0.000</td>
      <td>0.764</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>@BBC</td>
      <td>RT @bbcweather: #Winter may not be done with u...</td>
      <td>Sun Mar 25 08:43:26 +0000 2018</td>
      <td>-0.4019</td>
      <td>0.000</td>
      <td>0.101</td>
      <td>0.899</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>@BBC</td>
      <td>RT @BBCSport: A simply astonishing confession ...</td>
      <td>Sun Mar 25 08:43:22 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>@BBC</td>
      <td>A woman who drinks 30 cans a day says her addi...</td>
      <td>Sun Mar 25 08:00:12 +0000 2018</td>
      <td>-0.2500</td>
      <td>0.104</td>
      <td>0.153</td>
      <td>0.743</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>@BBC</td>
      <td>You learn something new every day. Here's how ...</td>
      <td>Sun Mar 25 07:34:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>@BBC</td>
      <td>When the clocks have gone forward but there's ...</td>
      <td>Sun Mar 25 07:02:00 +0000 2018</td>
      <td>-0.4215</td>
      <td>0.000</td>
      <td>0.141</td>
      <td>0.859</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>@BBC</td>
      <td>"I still see people screaming for help."\n\nTh...</td>
      <td>Sat Mar 24 20:34:01 +0000 2018</td>
      <td>-0.3818</td>
      <td>0.000</td>
      <td>0.115</td>
      <td>0.885</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>@BBC</td>
      <td>The story of the last decade of Picasso's life...</td>
      <td>Sat Mar 24 20:00:11 +0000 2018</td>
      <td>0.4767</td>
      <td>0.134</td>
      <td>0.000</td>
      <td>0.866</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>@BBC</td>
      <td>Could this be an answer to global water shorta...</td>
      <td>Sun Mar 25 19:44:01 +0000 2018</td>
      <td>0.1280</td>
      <td>0.101</td>
      <td>0.077</td>
      <td>0.821</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>@BBC</td>
      <td>Tonight, @regyates meets people whose lives ha...</td>
      <td>Sun Mar 25 19:15:06 +0000 2018</td>
      <td>-0.7506</td>
      <td>0.000</td>
      <td>0.286</td>
      <td>0.714</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>@BBC</td>
      <td>Tonight, @mcgregor_ewan and @McgColin celebrat...</td>
      <td>Sun Mar 25 18:40:04 +0000 2018</td>
      <td>0.5719</td>
      <td>0.163</td>
      <td>0.000</td>
      <td>0.837</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>@BBC</td>
      <td>The first ever statue of David Bowie has been ...</td>
      <td>Sun Mar 25 18:13:03 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>@BBC</td>
      <td>When you're enjoying being single and people j...</td>
      <td>Sun Mar 25 17:30:07 +0000 2018</td>
      <td>0.5267</td>
      <td>0.185</td>
      <td>0.000</td>
      <td>0.815</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>@BBC</td>
      <td>🇺🇸🏝🇬🇧 Welcome to Tangier Island, the tiny US i...</td>
      <td>Sun Mar 25 15:03:03 +0000 2018</td>
      <td>0.4588</td>
      <td>0.167</td>
      <td>0.000</td>
      <td>0.833</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>@BBC</td>
      <td>💬 We could listen to him speak all day. \n\n📽 ...</td>
      <td>Sun Mar 25 14:30:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>@BBC</td>
      <td>Predictions suggest a build-up of about 80,000...</td>
      <td>Sun Mar 25 14:09:03 +0000 2018</td>
      <td>0.1779</td>
      <td>0.091</td>
      <td>0.000</td>
      <td>0.909</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>@BBC</td>
      <td>👽✨🛸 @prattprattpratt stars as a happy-go-lucky...</td>
      <td>Sun Mar 25 13:03:04 +0000 2018</td>
      <td>0.5574</td>
      <td>0.184</td>
      <td>0.000</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>@BBC</td>
      <td>Weighing just 100g, a newborn panda is one 900...</td>
      <td>Sun Mar 25 12:03:02 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>71</td>
      <td>@nytimes</td>
      <td>"Most teenagers talk about drama about girlfri...</td>
      <td>Sun Mar 25 13:45:06 +0000 2018</td>
      <td>-0.7184</td>
      <td>0.000</td>
      <td>0.273</td>
      <td>0.727</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>72</td>
      <td>@nytimes</td>
      <td>The first American woman to win an Olympic cha...</td>
      <td>Sun Mar 25 13:30:11 +0000 2018</td>
      <td>0.4767</td>
      <td>0.265</td>
      <td>0.142</td>
      <td>0.593</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>73</td>
      <td>@nytimes</td>
      <td>To many in Washington, Stephanie Clifford, bet...</td>
      <td>Sun Mar 25 13:00:09 +0000 2018</td>
      <td>0.4404</td>
      <td>0.132</td>
      <td>0.000</td>
      <td>0.868</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>74</td>
      <td>@nytimes</td>
      <td>China Splits Top Jobs at Central Bank, Adding ...</td>
      <td>Sun Mar 25 12:49:37 +0000 2018</td>
      <td>0.2023</td>
      <td>0.153</td>
      <td>0.000</td>
      <td>0.847</td>
    </tr>
    <tr>
      <th>1474</th>
      <td>75</td>
      <td>@nytimes</td>
      <td>"We’re going to be the generation that takes d...</td>
      <td>Sun Mar 25 12:45:06 +0000 2018</td>
      <td>-0.3400</td>
      <td>0.000</td>
      <td>0.167</td>
      <td>0.833</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>76</td>
      <td>@nytimes</td>
      <td>One Houston suburb devastated by Harvey is sur...</td>
      <td>Sun Mar 25 12:30:02 +0000 2018</td>
      <td>-0.7964</td>
      <td>0.083</td>
      <td>0.303</td>
      <td>0.614</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>77</td>
      <td>@nytimes</td>
      <td>Using Digital Firm, Brexit Campaigners Skirted...</td>
      <td>Sun Mar 25 12:10:19 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>78</td>
      <td>@nytimes</td>
      <td>Slammed in September by Hurricanes Irma and Ma...</td>
      <td>Sun Mar 25 12:05:47 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1478</th>
      <td>79</td>
      <td>@nytimes</td>
      <td>There was no way around the buzz-kill query: H...</td>
      <td>Sun Mar 25 11:46:37 +0000 2018</td>
      <td>-0.5423</td>
      <td>0.000</td>
      <td>0.184</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>80</td>
      <td>@nytimes</td>
      <td>Internet companies were built on a model in wh...</td>
      <td>Sun Mar 25 11:30:08 +0000 2018</td>
      <td>0.5106</td>
      <td>0.148</td>
      <td>0.000</td>
      <td>0.852</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>81</td>
      <td>@nytimes</td>
      <td>Photos from the #MarchForOurLives protests aro...</td>
      <td>Sun Mar 25 11:15:14 +0000 2018</td>
      <td>-0.2263</td>
      <td>0.000</td>
      <td>0.174</td>
      <td>0.826</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>82</td>
      <td>@nytimes</td>
      <td>Washington is now consumed by a debate over wh...</td>
      <td>Sun Mar 25 11:00:19 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>83</td>
      <td>@nytimes</td>
      <td>In one of Sweden's gender-free schools, a girl...</td>
      <td>Sun Mar 25 10:30:13 +0000 2018</td>
      <td>-0.5994</td>
      <td>0.000</td>
      <td>0.151</td>
      <td>0.849</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>84</td>
      <td>@nytimes</td>
      <td>City Kitchen: A Crisp Cool-Weather Twist for a...</td>
      <td>Sun Mar 25 10:01:34 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1484</th>
      <td>85</td>
      <td>@nytimes</td>
      <td>The U.S. and China are waging a cold war for g...</td>
      <td>Sun Mar 25 10:00:11 +0000 2018</td>
      <td>-0.4767</td>
      <td>0.073</td>
      <td>0.158</td>
      <td>0.769</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>86</td>
      <td>@nytimes</td>
      <td>Greenhouse Gas Emissions Rose Last Year. Here ...</td>
      <td>Sun Mar 25 09:32:47 +0000 2018</td>
      <td>0.2023</td>
      <td>0.141</td>
      <td>0.000</td>
      <td>0.859</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>87</td>
      <td>@nytimes</td>
      <td>“How can this be? How can we have so much info...</td>
      <td>Sun Mar 25 09:25:03 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>88</td>
      <td>@nytimes</td>
      <td>There are substantial coverage gaps in traditi...</td>
      <td>Sun Mar 25 09:03:48 +0000 2018</td>
      <td>0.6124</td>
      <td>0.250</td>
      <td>0.000</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>89</td>
      <td>@nytimes</td>
      <td>Where do these Mets players go when they're hu...</td>
      <td>Sun Mar 25 08:47:20 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>90</td>
      <td>@nytimes</td>
      <td>You'll probably need to do a little shopping t...</td>
      <td>Sun Mar 25 08:31:06 +0000 2018</td>
      <td>0.4588</td>
      <td>0.200</td>
      <td>0.000</td>
      <td>0.800</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>91</td>
      <td>@nytimes</td>
      <td>Sharing is caring. It's also become a trend at...</td>
      <td>Sun Mar 25 08:14:54 +0000 2018</td>
      <td>0.5565</td>
      <td>0.280</td>
      <td>0.112</td>
      <td>0.607</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>92</td>
      <td>@nytimes</td>
      <td>Can Facebook be fixed? https://t.co/0BKiz8K04B</td>
      <td>Sun Mar 25 07:58:14 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>93</td>
      <td>@nytimes</td>
      <td>Sketch Guy: Resistance Is Futile. To Change Ha...</td>
      <td>Sun Mar 25 07:39:33 +0000 2018</td>
      <td>-0.4404</td>
      <td>0.000</td>
      <td>0.209</td>
      <td>0.791</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>94</td>
      <td>@nytimes</td>
      <td>Greenhouse gas emissions rose last year. Here ...</td>
      <td>Sun Mar 25 07:34:28 +0000 2018</td>
      <td>0.2023</td>
      <td>0.141</td>
      <td>0.000</td>
      <td>0.859</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>95</td>
      <td>@nytimes</td>
      <td>The Great Australian Bight, a pristine stretch...</td>
      <td>Sun Mar 25 07:16:06 +0000 2018</td>
      <td>0.7650</td>
      <td>0.280</td>
      <td>0.000</td>
      <td>0.720</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>96</td>
      <td>@nytimes</td>
      <td>The Trump administration raised fears of a tra...</td>
      <td>Sun Mar 25 07:00:25 +0000 2018</td>
      <td>-0.7717</td>
      <td>0.000</td>
      <td>0.295</td>
      <td>0.705</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>97</td>
      <td>@nytimes</td>
      <td>In a story with parallels to that of the Ameri...</td>
      <td>Sun Mar 25 06:42:09 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>98</td>
      <td>@nytimes</td>
      <td>Is it worth making pita at home? Absolutely. h...</td>
      <td>Sun Mar 25 06:23:45 +0000 2018</td>
      <td>-0.3612</td>
      <td>0.154</td>
      <td>0.276</td>
      <td>0.569</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>99</td>
      <td>@nytimes</td>
      <td>How do you remove 200,000 pounds of trash from...</td>
      <td>Sun Mar 25 06:05:15 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>100</td>
      <td>@nytimes</td>
      <td>There will always be something gray and Dicken...</td>
      <td>Sun Mar 25 05:46:35 +0000 2018</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 8 columns</p>
</div>




```python
#plt.scatter(tweets_df['Twitter Name'], tweets_df['Compound Score'], tweets_df['count'])
```


```python
bbc = tweets_df.loc[tweets_df['Twitter Name'] == '@BBC']
cbs = tweets_df.loc[tweets_df['Twitter Name'] == '@CBS']
cnn = tweets_df.loc[tweets_df['Twitter Name'] == '@CNN']
fox = tweets_df.loc[tweets_df['Twitter Name'] == '@FoxNews']
nyc = tweets_df.loc[tweets_df['Twitter Name'] == '@nytimes']
```


```python
plt.scatter(bbc['Tweets Ago'],bbc['Compound Score'], color='black', label="BBC")
plt.scatter(cbs['Tweets Ago'],cbs['Compound Score'], color='blue', label="BBC")
plt.scatter(cnn['Tweets Ago'],cnn['Compound Score'], color='red', label="BBC")
plt.scatter(fox['Tweets Ago'],fox['Compound Score'], color='orange', label="BBC")
plt.scatter(nyc['Tweets Ago'],nyc['Compound Score'], color='green', label="BBC")
plt.xlabel('Tweets Ago')
plt.ylabel("Sentiment Compound Score")
plt.title('Twitter News Sentiment Analysis')
#plt.legend()
#plt.legend would not seperate out plot....
plt.show()
plt.savefig('Scatter.png')
```


![png](output_11_0.png)



    <matplotlib.figure.Figure at 0x10a81e4a8>



```python
a = bbc['Compound Score'].mean()
b = cbs['Compound Score'].mean()
c = cnn['Compound Score'].mean()
d = fox['Compound Score'].mean()
e = nyc['Compound Score'].mean()
```


```python
plt.bar("bbc", a, color = "black")
plt.bar("cbs", b , color = "blue")
plt.bar("cnn", c, color = "red")
plt.bar("fox", d, color = "yellow")
plt.bar("NY Times", e, color = "green")
plt.ylabel("Sentiment Compound Score")
plt.title("Compound mean Scores")
plt.show()
plt.savefig('Analysis_Mean.png')
```


![png](output_13_0.png)



    <matplotlib.figure.Figure at 0x10a8d38d0>

