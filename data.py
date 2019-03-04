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
from itertools import dropwhile

def get_data():
    # funksjon som tilordner innholdet i filen train.csv til en variabel,
    # og leser teksten til en ny variabel text. Returnerer verdien til variabelen text.
    # Legger så til ord fra variabelen med tekst til en liste train_text, hvis ordet ikke allerede befinner seg i listen.
    path = 'train.csv'
    data = pd.read_csv(path)
    text = data['text']

    train_text = []
    for i in range(len(text)):
        if text[i] not in train_text:
            train_text.append(text[i])

    return train_text


def clean_data(text):
    # Funksjon hvor regex blir brukt til å fjerne alle bokstaver som ikke er i det engelske alfabetet.
    # Filtrerer bort stopwords som this, and, or osv.

    text = re.sub("[^ A-Za-z]", '', str(text)).lower().split()

    stop_words = set(stopwords.words('english'))
    text = [w for w in text if w not in stop_words]
#    text = str(text).strip().split()

    return text


def count_data(text):
    # Funksjon som teller de mest vanlige ordene som dukker opp i datasettet
    # og skriver dem ut til en ny fil 'vocabulary.csv', hvis filen ikke eksisterer,
    # eller overskriver hvis den eksisterer
    # repr() metoden returnerer en utskrivbar representasjonsstreng av variabelen
    text = clean_data(text)
    c = Counter(text)
    count = Counter(k for k in c.elements() if c[k] >= 5)

    repr(count)
#     with open('vocabulary.txt', 'w+') as f:
    with open('vocabulary.csv', 'w+') as f:
        f.write(repr(count) + '\n')
        f.close()
    print(count)

    return count


def run():
    # Her kjøres programmet. Dataen blir hentet fra csv filen, og filtreres.
    # Returnerer et vokabular med de mest nevnte ordene.
    count_data(get_data())


# tilkaller funksjonen som kjører programmet
run()
