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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy import spatial

df = ""


def helper(x):
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


def kmeans():
    # cluster the data
    X = df["vector"].values
    X = np.stack(X, axis=0)
    # km = KMeans(n_clusters=8, init='k-means++')
    km = KMeans(n_clusters=8)
    km = km.fit(X)
    centroids = km.cluster_centers_

    cluster_map = pd.DataFrame()
    # cluster_map['ids'] = df.rid.values
    cluster_map['reviews'] = df['ustories']
    cluster_map['vector'] = df.vector.values
    cluster_map['cluster'] = km.labels_
    cluster_map['stories'] = df['stories']
    # cluster_map['novelty'] = df.novelty_avg.values

    for i in range(0, 8):
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
            if sse > 0.98:
                print(row['reviews'])
            cosine_distances.append(sse)
        print(collections.Counter(words).most_common(10))
        print("*********************")


def dbscan():
    X = df["vector"].values
    X = np.stack(X, axis=0)
    dbs = DBSCAN(eps=0.9, min_samples=2, metric='cosine').fit(X)
    print("number of labels:", collections.Counter(dbs.labels_))
    # print(clustering.labels_)
    cluster_map = pd.DataFrame()
    # cluster_map['ids'] = df.rid.values
    cluster_map['reviews'] = df['stories']
    cluster_map['vector'] = df.vector.values
    cluster_map['cluster'] = dbs.labels_
    cluster_map['stories'] = df['stories']
    # cluster_map['novelty'] = df.novelty_avg.values

    words = []
    print("number of points in cluster", len(cluster_map[cluster_map.cluster
                                                         == -1]))
    for _, row in cluster_map[cluster_map.cluster == -1].iterrows():
        words = words + row['stories']
        print(row['reviews'])
    print(collections.Counter(words).most_common(10))
    print("*********************")


def main():
    global df
    start_time = time.time()
    # df = pd.read_csv('preprocessed_text_temp.csv')
    df = pd.read_csv('preprocessed_text_hr_temp.csv')
    # df = pd.read_csv('preprocessed_text_alexa_temp.csv', delimiter='\t')
    # df['storiescopy'] = df['stories']
    # print(df['stories'])
    # df.dropna(inplace=True)
    print("--- %s seconds to read the file ---" % (time.time() -
                                                   start_time))

    vector(tf_idf())
    kmeans()
    dbscan()


if __name__ == '__main__':
    main()