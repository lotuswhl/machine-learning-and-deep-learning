import pandas as pd
import numpy as np
import nltk
import itertools
import os

"""
preprocess the csv file :reddit-comments-2015-08.csv
    the file contains comments on reddit of each row
preprocess as follows:
    - read csv file with pandas
    - convert the comments dtype to string (default is object type of pandas) and to lower case
    - convert each comments to sentences with nltk.sent_tokenize method (since in English,sentences can be split with "."," ?","!"...etc. It would be complicated if we handle these on our own. Luckily ,with nltk ,we can handle this easily. )
    - tokenize the sentences into words with nltk.word_tokenize method
    - append "sentence_start" flag at the begining of each sentence and "sentence_end" at the end of each senetce,since we will need to predict from the model ,(aka,generate text) ; So we need to predict the flag as well.
"""

data_path = os.path.join(os.getcwd(),"data","reddit-comments-2015-08.csv")

SENTENCE_START = "senentence_start"
SENTENCE_END = "sentence_end"


def get_raw_data(data_path):
    df = pd.read_csv(data_path)
    comments = df.body.astype(str)
    return comments.values


def convert_data_2_sentences(data_strs):
    sentences = []
    for sents in data_strs:
        sentences.extend(nltk.sent_tokenize(sents.lower()))
    return sentences

def convert_sentences_2_words(sentences):
    word_sents = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        words.insert(0,SENTENCE_START)
        words.append(SENTENCE_END)
        word_sents.append(words)
    return word_sents

def get_tokenized_sentences():
    return convert_sentences_2_words(convert_data_2_sentences(get_raw_data(data_path)))






def main():
    tokenized_sents=get_tokenized_sentences()
    print(tokenized_sents[:3])


if __name__ == '__main__':
    main()
        