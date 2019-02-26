import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from scipy import sparse
from IPython.display import display
import re
import string
from collections import Counter
import csv
import pickle

def get_data():
    # Gets the data from a path, at this moment 'train.csv'
    # file must be in the same folder as data.py
    path = 'train.csv'
    data = pd.read_csv(path)
    text = data['text']

    train_text = []
    for i in range(len(text)):
        if text[i] not in train_text:
            train_text.append(text[i])

    return train_text


def clean_data(text):
    # Cleans up the data for unwanted characters

    text = re.sub("[^ A-Za-z0-9]", '', str(text)).lower().split()

    stop_words = set(stopwords.words('english'))
    text = [w for w in text if w not in stop_words]
    text = str(text).strip().split()

    return text


def count_data(text):
    # Counts the most common words that occurs in the data
    # and writes them to a new vocabulary file

    text = clean_data(text)
    count = Counter(text).most_common()
    f = open('vocabulary.txt', 'wb+')
    pickle.dump(count, f)
    f.close()
    print(count)

    return count




def run():
    # Function that runs the code
    count_data(get_data())


run()
