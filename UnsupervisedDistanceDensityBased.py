__author__ = "Kirtana Suresh"

"""
File: UnsupervisedDistanceDensityBased.py

Author: Kirtana Suresh <ks3057@rit.edu>

Course: SWEN 789 01

Description:
Contains unsupervised methods for Novelty Detection:
KMeans, DBSCAN, Mean Shift

According to workers, the number of novel ideas in DB:
0    1888
1     995

where 0 = novelty rating < 4 and 1 >= 4
"""


import pandas as pd
import time
import math
import collections
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy import spatial
import itertools
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import argparse


df = ""
f = open("unsupervised_novelty.txt", "w")
calculated_novelty = set()


def cmdline_input():
    """
    Takes the command line input from the user as program arguments.
    :return: A Namespace object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--calculateEpsilon', action='store_true')

    parser.add_argument('--meanShift', action='store_true')

    args = parser.parse_args()

    return args


def helper(x):
    """
    Helper function for TF
    :param x: user stories to be split
    :return: list of user words
    """
    try:
        return x.split()
    except:
        return


def tf():
    """
    Calculates TF
    :return: None
    """
    # df['stories'] = df['stories'].apply(lambda x: x.split())
    df['stories'] = df['stories'].apply(lambda x: helper(x))
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
    """
    Sums IDF of each document
    :return: IDF dict
    """
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


def vector_helper(word, d):
    """
    Helper for the vector function. Check if word is present in the
    dictionary, if yes, return its tf-idf value, or return 0
    :param word: word to be checked
    :param d: dictionary containing tf idf values
    :return: value of the word (either tfidf value or 0)
    """

    if word in d:
        return d[word]
    else:
        return 0


def vector(idf_dict):
    """
    Converts all documents to vectors
    :param idf_dict: the idf dictionary
    :return: None
    """
    df['vector'] = df['tfidf'].apply(lambda d: np.array([vector_helper(word, d)
                        for word, idfval in idf_dict.items()]))


def classify():
    """
    0 = novelty rating < 4 and 1 >= 4
    :return: None
    """
    df['class'] = df['novelty_avg'].apply(lambda v: 1 if v >= 4 else 0)


def kmeans():
    """
    Usupervised K Means ++ Novelty Detection
    :return:
    """
    # cluster the data
    k = 8
    X = df["vector"].values
    X = np.stack(X, axis=0)
    # km = KMeans(n_clusters=8, init='k-means++')
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    centroids = km.cluster_centers_

    cluster_map = pd.DataFrame()
    cluster_map['rid'] = df.rid.values
    cluster_map['reviews'] = df['ustories']
    cluster_map['vector'] = df.vector.values
    cluster_map['cluster'] = km.labels_
    cluster_map['stories'] = df['stories']

    silhouette_avg = silhouette_score(X, km.labels_)
    print("For n_clusters =", len(km.labels_),
          "The average silhouette_score is :", silhouette_avg)

    for i in range(0, k):
        cosine_distances = []
        words = []
        length_of_cluster = len(cluster_map[cluster_map.cluster == i])
        print("*********************")
        print("number of points in cluster", length_of_cluster)
        if length_of_cluster == 1:
            print(cluster_map[cluster_map.cluster == i]['reviews'])

        for _, row in cluster_map[cluster_map.cluster == i].iterrows():
            sse = spatial.distance.cosine(row['vector'], centroids[i])
            words = words + row['stories']
            if sse > 0.9:
                f.write(row['reviews'])
                f.write('\n')
                calculated_novelty.add(row['rid'])
            cosine_distances.append(sse)
        print(collections.Counter(words).most_common(10))

    match = set(df['rid'].loc[df['class'] == 1]).intersection(
        calculated_novelty)
    print("accuracy:", 100 * len(match) / 995)


def dbscan():
    """
    Usupervised DBSCAN Novelty Detection
    :return:
    """
    X = df["vector"].values
    X = np.stack(X, axis=0)
    dbs = DBSCAN(eps=0.9, min_samples=21, metric='cosine').fit(X)
    labels = collections.Counter(dbs.labels_)

    silhouette_avg = silhouette_score(X, dbs.labels_)
    print("For n_clusters =", len(labels),
          "The average silhouette_score is :", silhouette_avg)

    print("number of labels:", collections.Counter(dbs.labels_))
    cluster_map = pd.DataFrame()
    cluster_map['stories'] = df['stories']
    cluster_map['vector'] = df.vector.values
    cluster_map['cluster'] = dbs.labels_
    cluster_map['ustories'] = df['ustories']

    for key in labels.keys():
        if key != -1:
            list1 = cluster_map['stories'].loc[cluster_map['cluster'] ==
                                             key].tolist()
            list2 = list(itertools.chain.from_iterable(list1))
            print(collections.Counter(list2).most_common(10))
            print("*********************")

    words = []
    print("number of points in cluster", len(cluster_map[cluster_map.cluster
                                                         == -1]))
    for _, row in cluster_map[cluster_map.cluster == -1].iterrows():
        words = words + row['stories']
        print(row['ustories'])
        f.write(row['ustories'])
        f.write('\n')
    print(collections.Counter(words).most_common(10))
    print("*********************")


def meanshift():
    """
    Usupervised Mean Shift Novelty Detection
    :return:
    """
    X = df["vector"].values
    X = np.stack(X, axis=0)
    ms = MeanShift().fit(X)
    labels = collections.Counter(ms.labels_)
    print("number of labels:", labels)
    silhouette_avg = silhouette_score(X, ms.labels_)
    print("For n_clusters =", len(labels),
          "The average silhouette_score is :", silhouette_avg)

    cluster_map = pd.DataFrame()
    cluster_map['reviews'] = df['stories']
    cluster_map['vector'] = df.vector.values
    cluster_map['cluster'] = ms.labels_
    cluster_map['ustories'] = df['ustories']
    cluster_map['rid'] = df['rid']

    cn = set()
    for key in labels.keys():
        length_of_cluster = len(cluster_map[cluster_map.cluster == key])
        if length_of_cluster == 1:
            cn.add(cluster_map[cluster_map.cluster == key]['rid'].iloc[0])
            f.write(cluster_map[cluster_map.cluster == key]['ustories'].iloc[0])
            f.write('\n')
        else:
            list1 = cluster_map['reviews'].loc[cluster_map['cluster'] ==
                                               key].tolist()
            list2 = list(itertools.chain.from_iterable(list1))
            print(collections.Counter(list2).most_common(10))
            print("*********************")

    match = set(df['rid'].loc[df['class'] == 1]).intersection(
        calculated_novelty)
    print("accuracy:", 100 * len(match) / 995)


def calculate_epsilon():
    """
    Calculate epsilon for DBSCAN
    :return:
    """
    X = df["vector"].values
    X = np.stack(X, axis=0)
    ns = 22
    nbrs = NearestNeighbors(n_neighbors=ns, metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    distanceDec = sorted(distances[:, ns - 1], reverse=True)
    plt.plot(list(range(1, len(df) + 1)), distanceDec)
    plt.grid()
    plt.show()


def main():
    """
    Responsible for handling all function calls
    :return: None
    """
    global df
    start_time = time.time()
    df = pd.read_csv('preprocessed_text_temp.csv')
    print("--- %s seconds to read the file ---" % (time.time() -
                                                   start_time))

    classify()
    vector(tf_idf())
    args = cmdline_input()
    if args.calculateEpsilon:
        calculate_epsilon()
    kmeans()
    dbscan()
    if args.meanShift:
        meanshift()


if __name__ == '__main__':
    main()