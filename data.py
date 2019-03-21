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
from collections import Counter
from nltk.classify import NaiveBayesClassifier
import string
import random
import math


def split_data(data, split):
    train_size = int(len(data) * split)
    train_set = []
    copy = list(data)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def get_text(path):
    # funksjon som tilordner innholdet i filen train.csv til en variabel,
    # og leser teksten til en ny variabel text. Returnerer verdien til variabelen text.
    # Legger så til ord fra variabelen med tekst til en liste train_text, hvis ordet ikke allerede befinner seg i listen.
    news = pd.read_csv(path, encoding='utf-8')
    print(".............Getting news-articles.............")
    text = news.text
    print(len(text))
    train_text = []
    for i in range(len(text)):
        if text[i] not in train_text:
            train_text.append(text[i])

    return train_text


def clean_text(text):
    # Funksjon hvor regex blir brukt til å fjerne alle bokstaver som ikke er i det engelske alfabetet.
    # Filtrerer bort stopwords som this, and, or osv.
    print(".............Stripping and cleaning the text down to words.............")
    text = re.sub("[^ A-Za-z]", '', str(text)).lower().split()

    stop_words = set(stopwords.words('english'))
    text = [w for w in text if w not in stop_words]
#    text = str(text).strip().split()

    return text


def create_vocabulary(text):
    # Funksjon som teller de mest vanlige ordene som dukker opp i datasettet
    # og skriver dem ut til en ny fil 'vocabulary.csv', hvis filen ikke eksisterer,
    # eller overskriver hvis den eksisterer
    # repr() metoden returnerer en utskrivbar representasjonsstreng av variabelen
    print(".............Counting all occurances of words.............")
    text = clean_text(text)
    c = Counter(text)
    count = Counter(k for k in c.elements() if c[k] >= 100)
    print(".............Creating a vocabulary.............")
    repr(count)
#     with open('vocabulary.txt', 'w+') as f:
    with open('vocabulary.csv', 'w+') as f:
        f.write(repr(count) + '\n')
        f.close()
        print(".............Vocabulary has been created.............")
    return count


def run():
    # Her kjøres programmet. Dataen blir hentet fra csv filen, og filtreres.
    # Returnerer et vokabular med de mest nevnte ordene.

    path = 'train.csv'
    news = get_text(path)
    create_vocabulary(news)


# tilkaller funksjonen som kjører programmet
run()


class MNBclassifier(object):

    def clean_news(self, text):
        # Funksjon hvor regex blir brukt til å fjerne alle bokstaver som ikke er i det engelske alfabetet.
        # Filtrerer bort stopwords som this, and, or osv.
        print(".............Stripping and cleaning the text down to words.............")
        text = re.sub("[^ A-Za-z]", '', str(text)).lower().split()

        stop_words = set(stopwords.words('english'))
        text = [w for w in text if w not in stop_words]
        #    text = str(text).strip().split()

        return text

    def count_data(self, words):

        #word_counts = Counter(words)
        #word_counts = Counter(k for k in word_counts.elements() if word_counts[k] >= 100)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        return word_counts

    def fitting(self, news, labels):
        self.vocabulary = pd.read_csv('vocabulary.csv', delimiter=',')
        self.vocab = self.vocabulary.copy()
        self.num_articles = {}
        self.log_class_priors = {}
        self.word_counts = {}

        self.num_articles['fake'] = sum(1 for label in labels if label == 1)
        self.num_articles['real'] = sum(1 for label in labels if label == 0)
        self.log_class_priors['fake'] = math.log(self.num_articles['fake'] / len(X))
        self.log_class_priors['real'] = math.log(self.num_articles['real'] / len(X))
        self.word_counts['fake'] = {}
        self.word_counts['real'] = {}

        for x, y in zip(news, labels):
            C = 'fake' if y == 1 else 'real'
            counts = self.count_data(self.clean_news(x))

            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[C]:
                    self.word_counts[C][word] = 0

                self.word_counts[C][word] += count
        print(len(self.word_counts['real']))
        print(len(self.word_counts['fake']))
        print("there is {} articles in fake news and {} articles in real news".format(self.num_articles['fake'],
                                                                                      self.num_articles['real']))


if __name__ == '__main__':

    path = 'train.csv'

    news2 = pd.read_csv(path, encoding='utf-8')
    X = news2.text
    y = news2.label

    # split = 0.67
    # X_train, X_test = split_data(X, split)
    # y_train, y_test = split_data(y, split)
    # print('Split {0} articles where {1} articles is for training and {2} articles is for testing'.format(len(X),
    #                                                                                                      len(X_train),
    #                                                                                                      len(X_test)))
    MNB = MNBclassifier()
    MNB.fitting(X, y)

    # pred = MNB.predict(X)
    # true = y
    #
    # accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i] / float(len(pred)))
    # print("Accuracy: {:.001%}".format(accuracy))
