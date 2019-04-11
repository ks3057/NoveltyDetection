__author__ = "Kirtana Suresh"

"""
File: Supervised.py

Author: Kirtana Suresh <ks3057@rit.edu>

Course: SWEN 789 01

Description:
Supervised classification of Novelty

According to workers, the number of novel ideas in DB:
0    1888
1     995

where 0 = novelty rating < 4 and 1 >= 4
"""


import pandas as pd
import time
import math
import collections
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = ""


def tf():
    """
    Calculates TF
    :return: None
    """
    # get frequency of all words in each document
    df['tf'] = df['stories'].apply(collections.Counter)

    # returns dictionary where word is key and tf is value
    df['tf'] = df['tf'].apply(lambda d: {word: count / sum(d.values())
                                         for word, count in d.items()})


def idf():
    """
    Calculate Inverse Document Frequency
    :return: None
    """
    count_dict = {}

    # count of the number of stories each word appears in
    for tf_dict in df['tf']:
        for word in tf_dict:
            if word in count_dict:
                count_dict[word] += 1
            else:
                count_dict[word] = 1

    # calculate idf dictionary
    length_df = len(df)
    idf_dict = {}
    for word in count_dict:
        idf_dict[word] = math.log(length_df / count_dict[word])
    return idf_dict


def idf_col():
    idf_dict = idf()
    df['idf'] = df['tf'].apply(lambda d: {word: idf_dict[word]
                                         for word in d.keys()})

    # calculate sum of idf
    df['sumidf'] = df['idf'].apply(lambda d: sum(d.values()))

    return idf_dict


def tf_idf():
    """
    Calculate TF IDF values
    :return: Idf dictionary for calculating the vector values
    """
    tf()
    idf_dict = idf_col()

    # calculate tfidf values. Key is word, value is its tf-idf
    df['tfidf'] = df['tf'].apply(lambda d: {word: tfval * idf_dict[word]
                                            for word, tfval in d.items()})

    # calculate sum of tfidf
    df['sumtfidf'] = df['tfidf'].apply(lambda d: sum(d.values()))

    return idf_dict


def classify():
    df['class'] = df['novelty_avg'].apply(lambda v: 1 if v >= 4 else 0)


def mnb():

    X_train, X_test, y_train, y_test = train_test_split(df, df['class'],
                                                        test_size=0.3)

    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    X_train = vectorizer.fit_transform(X_train['stories'])
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    print(classification_report(y_test, mnb.predict(vectorizer.transform(
        X_test['stories']))))


def dt():
    feature_cols = ['sumidf']
    X = df[feature_cols]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, clf.predict(X_test)))


def main():
    global df
    start_time = time.time()
    df = pd.read_csv('preprocessed_text.csv')
    print("--- %s seconds to read the file ---" % (time.time() -
                                                   start_time))

    tf_idf()
    classify()
    # mnb()
    dt()



if __name__ == '__main__':
    main()