import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
import re
from collections import Counter, defaultdict
import random
import math
import pickle
import csv


def split_data(data, split):
    # Funksjon som splitter datasettet med en angitt splitratio
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
    print("Total {} articles in the dataset".format(len(text)))
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

    word_counts = {}
    for word in text:
        word_counts[word] = word_counts.get(word, 0) + 1
    print("before", len(word_counts))

    for w in list(word_counts):
        if word_counts[w] < 100:
            del word_counts[w]

    print("after", len(word_counts))
    print(".............Creating a vocabulary.............")

    print("length of vocabulary", len(word_counts))
#     with open('vocabulary.txt', 'w+') as f:
    with open('vocabulary.csv', 'w+') as f:
        f.write(repr(word_counts) + '\n')
        f.close()

    print(".............Vocabulary has been created.............")
    return word_counts


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

        text = re.sub("[^ A-Za-z]", '', str(text)).lower().split()

        stop_words = set(stopwords.words('english'))
        text = [w for w in text if w not in stop_words]
        #    text = str(text).strip().split()

        return text

    def count_data(self, words):
        #Funksjon som teller antall ganger et ord oppstår

        #word_counts = Counter(words)
        #word_counts = Counter(k for k in word_counts.elements() if word_counts[k] >= 100)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0

        return word_counts

    def fitting(self, news, labels):
        #Funksjon som tilknytter training data mot classifier, og finner likelihood for at et ord enten er
        #reliable eller unreliable

        self.vocabulary = pd.read_csv('vocabulary.csv')
        self.vocab = self.vocabulary.columns.copy()

        self.num_articles = {}
        self.log_class_priors = {}
        self.word_counts = {}

        self.num_articles['fake'] = sum(1 for label in labels if label == 1)
        self.num_articles['real'] = sum(1 for label in labels if label == 0)
        self.log_class_priors['fake'] = math.log(self.num_articles['fake'] / len(news))
        self.log_class_priors['real'] = math.log(self.num_articles['real'] / len(news))
        self.word_counts['fake'] = {}
        self.word_counts['real'] = {}
        print(".............Stripping and cleaning the text down to words.............")
        for x, y in zip(news, labels):
            C = 'fake' if y == 1 else 'real'
            counts = self.count_data(self.clean_news(x))

            for word, count in counts.items():
                # if word not in self.vocab:
                #     continue
                if word not in self.word_counts[C]:
                    self.word_counts[C][word] = 0

                self.word_counts[C][word] += count

        for w in list(self.word_counts['fake']):
            if self.word_counts['fake'][w] < 100:
                del self.word_counts['fake'][w]
        print("length of fake count", len(self.word_counts['fake']))

        with open('label_one_count.csv', 'w+') as f:
            f.write(repr(self.word_counts['fake']) + '\n')
            f.close()

        for w in list(self.word_counts['real']):
            if self.word_counts['real'][w] < 100:
                del self.word_counts['real'][w]
        print("length of real count", len(self.word_counts['real']))

        with open('label_zero_count.csv', 'w+') as f:
            f.write(repr(self.word_counts['real']) + '\n')
            f.close()

        print("there is {} articles in fake news and {} articles in real news".format(self.num_articles['fake'],
                                                                                      self.num_articles['real']))

    def predict(self, news):

        result = []
        for x in news:
            counts = self.count_data(self.clean_news(x))
            fake_score = 0
            real_score = 0
            for word, _ in counts.items():
                if word not in self.vocab:
                    continue

                #  LapLace smoothing
                log_w_given_fake = math.log(
                    (self.word_counts['fake'].get(word, 0.0) + 1.0) / (self.num_articles['fake'] + len(self.vocab)))
                log_w_given_real = math.log(
                    (self.word_counts['real'].get(word, 0.0) + 1.0) / (self.num_articles['real'] + len(self.vocab)))

                fake_score += log_w_given_fake
                real_score += log_w_given_real

            fake_score += self.log_class_priors['fake']
            real_score += self.log_class_priors['real']
            print(real_score, fake_score)
            if fake_score > real_score:
                result.append(1)
            else:
                result.append(0)
            return result

if __name__ == '__main__':

    path = 'train.csv'

    news2 = pd.read_csv(path, encoding='utf-8')
    X = news2.text
    y = news2.label

    split = 0.67
    X_train, X_test = split_data(X, split)
    y_train, y_test = split_data(y, split)
    print('Split {0} articles where {1} articles is for training and {2} articles is for testing'.format(len(X),
                                                                                                         len(X_train),
                                                                                                         len(X_test)))
    MNB = MNBclassifier()
    print("..............Fitting Data..............")
    MNB.fitting(X_train, y_train)

    prediction = MNB.predict(X_train)
    true = y_train

    accuracy = sum(1 for i in range(len(prediction)) if prediction[i] == true[i] / float(len(prediction)))
    print("Accuracy: {:.0%}".format(accuracy))

    input_string = input('Please write something')
    input_news = MNB.count_data(clean_text(input_string))

    if MNB.predict(input_news) == 1:
        print('This is fake news')
    else:
        print('This is reliable news')

