__author__ = "Kirtana Suresh"

"""
File: TextPreprocess.py

Author: Kirtana Suresh <ks3057@rit.edu>

Course: SWEN 789 01

Description:
Text Preprocessing : lower case, tokenize, stop-word removal, pos tag and 
lemmatization
Output is saved to preprocessed_text.csv. 
"""


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import time
from langdetect import detect

df = ""
wordnet_lemmatizer = WordNetLemmatizer()


def lower_case():
    """
    Converts all documents to lower case
    :return: None
    """

    df['ustories'] = df['ustories'].str.lower()


def tokenize():
    """
    Tokenize all documents
    :return: None
    """
    df["stories"] = df["ustories"].apply(nltk.word_tokenize)


def stop_words():
    """
    Remove stop words from documents
    :return: None
    """
    stop_words_list = set(stopwords.words('english'))
    stop_words_list.add('smart')
    stop_words_list.add('home')
    stop_words_list.add('house')
    stop_words_list.add('smart-home')
    stop_words_list.add('automatic')
    stop_words_list.add('automatically')
    stop_words_list.add('n\'t')
    df['stories'] = df['stories'].apply(lambda li: list(filter(lambda word:
                                                               word not in
                                                               stop_words_list,
                                                               li)))


def pos_tags():
    """
    POS Tag all documents
    :return: None
    """
    df["stories"] = df["stories"].apply(nltk.pos_tag)

    pos_tags = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
                "VBZ", "RB", "RBR", "RBS", "WRB", "JJ", "JJR", "JJS"]
    df['stories'] = df['stories'].apply(lambda li: list(filter(lambda tup:
                                                               tup[1] in
                                                               pos_tags, li)))


def get_tag(word):
    """
    Helper function for lemmatizing. Returns the tag of the word to be
    lemmatized
    :param word: word to be lemmatized
    :return: tag of above word
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemma():
    """
    Lemmatize all documents
    :return: None
    """
    df['stories'] = df['stories'].apply(
        lambda li: [wordnet_lemmatizer.lemmatize(w, get_tag(w)) for
                    w, _ in li])

    df['stories'] = df['stories'].apply(' '.join)
    df.to_csv('preprocessed_text_hr_temp.csv', index=None,
              header=True)


def lang_detection():
    for index, row in df.iterrows():
        try:
            if detect(row['ustories']) != 'en':
                df.drop(index, inplace=True)
        except:
            print(row['ustories'])


def text_preprocess():
    """
    Calls all text preprocessing steps
    :return: None
    """
    lower_case()
    lang_detection()
    tokenize()
    stop_words()
    pos_tags()
    lemma()


def main():

    global df
    start_time = time.time()
    df = pd.read_csv('hotel_reviews.csv')
    # df = pd.read_csv('amazon_alexa.tsv', delimiter='\t')
    print("--- %s seconds to read the file ---" % (time.time() -
                                                   start_time))

    # df["ustories"] = df["role"] + " " + df["feature"] + " " + df["benefit"]
    # df["ustories"] = df["feature"] + " " + df["benefit"]

    start_time = time.time()
    text_preprocess()
    print("--- %s seconds for text preprocessing ---" % (time.time() -
                                                         start_time))


if __name__ == '__main__':
    main()