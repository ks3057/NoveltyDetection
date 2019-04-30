__author__ = "Kirtana Suresh"

"""
File: Supervised.py

Author: Kirtana Suresh <ks3057@rit.edu>

Course: SWEN 789 01

Description:
Supervised classification of Novelty.
Using classifiers Naive Bayes and Decision Tree
Using oversampling of minority class 
and AdaBoost if specified by user.

According to workers, the number of novel ideas in DB:
0    1888
1     995

where 0 = novelty rating < 4 and 1 >= 4
"""


import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as met
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import argparse

df = ""


def cmdline_input():
    """
    Takes the command line input from the user as program arguments.
    :return: A Namespace object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--boosting', action='store_true')

    args = parser.parse_args()

    return args


def classify():
    df['class'] = df['novelty_avg'].apply(lambda v: 1 if v >= 4 else 0)


def mnb():
    """
    Multinomial Naive Bayes Classifier using oversampling of minority class
    :return: None
    """

    skf = StratifiedKFold(n_splits=10)
    precision = 0
    recall = 0
    accuracy = 0
    f1_score = 0
    for train_index, test_index in skf.split(df, df['class']):
        train = df.loc[train_index.tolist(), :]
        X_train, X_test, y_train, y_test = train_test_split(train,
                                                            train['class'],
                                                            test_size=0.3)

        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        X_train = vectorizer.fit_transform(X_train['storiescopy'])

        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        mnb = MultinomialNB()
        mnb = mnb.fit(X_resampled, y_resampled)
        precision += met.precision_score(y_test, mnb.predict(
            vectorizer.transform(X_test['storiescopy'])))
        accuracy += met.accuracy_score(y_test, mnb.predict(
            vectorizer.transform(X_test['storiescopy'])))
        recall += met.recall_score(y_test, mnb.predict(
            vectorizer.transform(X_test['storiescopy'])))
        f1_score += met.f1_score(y_test, mnb.predict(
            vectorizer.transform(X_test['storiescopy'])))

    print("Naive Bayes Novelty Detection")
    print("precision%:", 100*precision/10, "recall%:", 100*recall/10)
    print("accuracy%:", 100*accuracy/10, "f1-score%", 100*f1_score/10)


def dt():
    """
    Decision Tree Classifier using oversampling of minority class
    :return: None
    """

    skf = StratifiedKFold(n_splits=10)
    precision = 0
    recall = 0
    accuracy = 0
    f1_score = 0
    for train_index, test_index in skf.split(df, df['class']):
        train = df.loc[train_index.tolist(), :]

        X_train, X_test, y_train, y_test = train_test_split(train,
                                                            train['class'],
                                                            test_size=0.3)

        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        X_train = vectorizer.fit_transform(X_train['storiescopy'])

        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        dt = DecisionTreeClassifier()
        dt = dt.fit(X_resampled, y_resampled)
        prediction = dt.predict(vectorizer.transform(X_test['storiescopy']))

        precision += met.precision_score(y_test, prediction)
        accuracy += met.accuracy_score(y_test, prediction)
        recall += met.recall_score(y_test, prediction)
        f1_score += met.f1_score(y_test, prediction)

    print("Decision Tree Novelty Detection")
    print("precision%:", 100 * precision / 10, "recall%:", 100 * recall / 10)
    print("accuracy%:", 100 * accuracy / 10, "f1-score%", 100 * f1_score / 10)


def adaboost():
    """
    Adaboost Ensemble Technique using Decision Tree as Base Estimator
    :return:
    """

    skf = StratifiedKFold(n_splits=10)
    precision = 0
    recall = 0
    accuracy = 0
    f1_score = 0
    for train_index, test_index in skf.split(df, df['class']):
        train = df.loc[train_index.tolist(), :]

        X_train, X_test, y_train, y_test = train_test_split(train,
                                                            train['class'],
                                                            test_size=0.3)

        vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        X_train = vectorizer.fit_transform(X_train['storiescopy'])

        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        dt = DecisionTreeClassifier()

        abc = AdaBoostClassifier(n_estimators=100, base_estimator=dt)
        abc = abc.fit(X_resampled, y_resampled)
        prediction = abc.predict(vectorizer.transform(X_test['storiescopy']))

        precision += met.precision_score(y_test, prediction)
        accuracy += met.accuracy_score(y_test, prediction)
        recall += met.recall_score(y_test, prediction)
        f1_score += met.f1_score(y_test, prediction)

    print("Adaboost Novelty Detection")
    print("precision%:", 100 * precision / 10, "recall%:", 100 * recall / 10)
    print("accuracy%:", 100 * accuracy / 10, "f1-score%", 100 * f1_score / 10)


def main():
    """
    Responsible for handling all function calls
    :return: None
    """
    global df
    start_time = time.time()
    df = pd.read_csv('preprocessed_text_temp.csv')
    df['storiescopy'] = df['stories']
    print("--- %s seconds to read the file ---" % (time.time() -
                                                   start_time))

    args = cmdline_input()

    classify()
    mnb()
    print()
    dt()
    print()
    if args.boosting:
        adaboost()


if __name__ == '__main__':
    main()