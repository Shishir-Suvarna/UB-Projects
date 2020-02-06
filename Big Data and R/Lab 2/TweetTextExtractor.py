import json
import os
import string

tweetsFolder = "tweets_DIC_LAB2"

sub_topics = ['Google', 'Amazon', 'Microsoft', 'Facebook', 'Apple']

for subtopic in sub_topics:
    folder = tweetsFolder + "\\" + subtopic
    count = 0
    for file in os.listdir(folder):
        if file.endswith(".json"):
            tweet_arr = []
            with open(folder + "\\" + file, "r") as f:
                tweets = json.load(f)
                for tweet in tweets:
                    tweet['text'] = " " .join([token for token in tweet['text'].split()
                                               if token.isalnum() and
                                               'http' not in token and
                                               '@' not in token and
                                               '<' not in token])
                    tweet_text = tweet['text'].translate(str.maketrans('', '', string.punctuation))
                    if " " + subtopic + " " not in tweet_text:
                        continue
                    count = +1
                    if tweet_text +"\n" not in tweet_arr:
                        tweet_arr.append(tweet_text + "\n")
                # count += len(tweets)

            with open(subtopic+"_1.txt", "a+", encoding='utf-8') as f:
                for tweet in tweet_arr:
                    f.write(tweet)

    print(subtopic + " count is: " + str(count))